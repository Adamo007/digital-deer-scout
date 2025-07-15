import streamlit as st
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import tempfile
import rasterio
import numpy as np
import os
from rasterio.mask import mask
from zipfile import ZipFile
import requests
from scipy.ndimage import gaussian_filter
import pydeck as pdk
from rasterio.transform import rowcol
from skimage.filters import sobel
import base64
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType
from scipy.ndimage import label, binary_erosion, binary_dilation
from sklearn.cluster import DBSCAN
import xml.etree.ElementTree as ET

st.set_page_config(page_title="Digital Deer Scout AI", layout="wide")
st.title("Digital Deer Scout - Terrain AI")

# Sidebar Controls
st.sidebar.header("Scouting Parameters")
wind = st.sidebar.selectbox("Wind Direction", ["NW", "W", "SW", "S", "SE", "E", "NE", "N"])
phase = st.sidebar.selectbox("Target Phase", ["Early Season", "Pre-Rut", "Rut"])
aggression = st.sidebar.slider("Pin Aggression Level", 1, 10, 5)
show_buck_beds = st.sidebar.checkbox("Show Buck Bedding", True)
show_doe_beds = st.sidebar.checkbox("Show Doe Bedding", True)
show_scrapes = st.sidebar.checkbox("Show Scrape Locations", True)
show_funnels = st.sidebar.checkbox("Show Funnels", True)
custom_tiff = st.sidebar.file_uploader("Upload GeoTIFF (DEM)", type=["tif", "tiff"])
custom_ndvi = st.sidebar.file_uploader("Upload NDVI GeoTIFF (optional)", type=["tif", "tiff"])

# KML/KMZ Extraction
def extract_kml(file) -> gpd.GeoDataFrame:
    try:
        if file.name.endswith(".kmz"):
            with ZipFile(file) as zf:
                kml_file = [f for f in zf.namelist() if f.endswith(".kml")]
                if not kml_file:
                    st.error("No KML file found in KMZ")
                    st.stop()
                with zf.open(kml_file[0]) as kmldata:
                    gdf = gpd.read_file(kmldata)
        else:
            gdf = gpd.read_file(file)
        if gdf.empty or not gdf.geometry.iloc[0].is_valid:
            st.error("Invalid or empty KML/KMZ geometry")
            st.stop()
        return gdf.to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"KML/KMZ parsing failed: {e}")
        st.stop()

# USGS DEM Fetch
def fetch_usgs_lidar(bounds, out_path="dem.tif"):
    minx, miny, maxx, maxy = bounds
    if (maxx - minx) > 1 or (maxy - miny) > 1:
        st.error("Bounding box too large for USGS DEM fetch")
        st.stop()
    url = (
        f"https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?bbox={minx},{miny},{maxx},{maxy}"
        "&bboxSR=4326&imageSR=4326&format=tiff&pixelType=F32&f=image"
    )
    r = requests.get(url)
    if r.status_code == 200 and r.headers.get("Content-Type") == "image/tiff":
        with open(out_path, "wb") as f:
            f.write(r.content)
        if os.path.getsize(out_path) < 2048:
            raise Exception("DEM download resulted in an empty file")
        return out_path
    raise Exception("USGS DEM download failed")

# NDVI Fetch using auto-refreshed token
def fetch_sentinel_ndvi(bounds, client_id, client_secret, out_path="ndvi.tif"):
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = "https://services.sentinel-hub.com"
    config.save()

    bbox = BBox(bbox=bounds, crs=CRS.WGS84)

    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B04", "B08"],
            output: { bands: 1 }
        };
    }
    function evaluatePixel(sample) {
        let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
        return [ndvi];
    }
    """

    try:
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A)
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=(512, 512),
            config=config,
            data_folder=tempfile.gettempdir()
        )
        data = request.get_data(save_data=True)
        if isinstance(data[0], np.ndarray):
            with rasterio.open(out_path, 'w', driver='GTiff', height=data[0].shape[0], width=data[0].shape[1], count=1,
                               dtype=str(data[0].dtype), crs='EPSG:4326', transform=rasterio.transform.from_bounds(*bounds, data[0].shape[1], data[0].shape[0])) as dst:
                dst.write(data[0], 1)
        else:
            with open(out_path, "wb") as f:
                f.write(data[0].read())
        return out_path
    except Exception as e:
        st.error(f"NDVI fetch failed: {e}")
        return None

# Terrain Analysis Functions
def analyze_terrain(dem_path, ndvi_path, bounds):
    """Analyze terrain for hunting features following Dan Infalt/Brad Herndon principles"""
    
    with rasterio.open(dem_path) as dem:
        elevation = dem.read(1)
        transform = dem.transform
        dem_shape = elevation.shape
        
    # Calculate slope and aspect
    dy, dx = np.gradient(elevation)
    slope = np.sqrt(dx**2 + dy**2)
    aspect = np.arctan2(dy, dx) * 180 / np.pi
    
    # Load NDVI if available and resample to match DEM
    ndvi = None
    if ndvi_path and os.path.exists(ndvi_path):
        try:
            with rasterio.open(ndvi_path) as ndvi_src:
                ndvi_data = ndvi_src.read(1)
                
                # If shapes don't match, resample NDVI to match DEM
                if ndvi_data.shape != dem_shape:
                    from rasterio.warp import reproject, Resampling
                    ndvi_resampled = np.empty(dem_shape, dtype=ndvi_data.dtype)
                    
                    reproject(
                        source=ndvi_data,
                        destination=ndvi_resampled,
                        src_transform=ndvi_src.transform,
                        src_crs=ndvi_src.crs,
                        dst_transform=transform,
                        dst_crs=dem.crs,
                        resampling=Resampling.bilinear
                    )
                    ndvi = ndvi_resampled
                else:
                    ndvi = ndvi_data
        except Exception as e:
            st.warning(f"NDVI processing failed: {e}")
            ndvi = None
    
    return {
        'elevation': elevation,
        'slope': slope,
        'aspect': aspect,
        'ndvi': ndvi,
        'transform': transform,
        'bounds': bounds
    }

def find_buck_bedding(terrain_data, wind_dir, aggression):
    """Find buck bedding areas - elevated, good visibility, thermal cover"""
    elevation = terrain_data['elevation']
    slope = terrain_data['slope']
    aspect = terrain_data['aspect']
    transform = terrain_data['transform']
    
    # More aggressive buck bedding criteria
    # 1. Elevated positions - lower threshold for more candidates
    valid_elevation = elevation[~np.isnan(elevation) & (elevation > 0)]
    if len(valid_elevation) == 0:
        return []
    
    # Use more lenient elevation threshold (top 50% instead of 30%)
    elevation_threshold = np.percentile(valid_elevation, 50)
    high_ground = elevation > elevation_threshold
    
    # 2. Broader slope range
    good_slope = (slope > 2) & (slope < 35)
    
    # 3. Enhanced thermal cover logic
    thermal_cover = np.ones_like(elevation, dtype=bool)
    if terrain_data['ndvi'] is not None:
        # More lenient NDVI threshold and handle edge cases
        ndvi_valid = ~np.isnan(terrain_data['ndvi'])
        thermal_cover = ndvi_valid & (terrain_data['ndvi'] > 0.2)
        
        # If no NDVI coverage, use elevation as proxy for cover
        if not np.any(thermal_cover):
            thermal_cover = elevation > np.percentile(valid_elevation, 40)
    else:
        # Without NDVI, use elevation variation as thermal cover proxy
        thermal_cover = elevation > np.percentile(valid_elevation, 40)
    
    # 4. More flexible wind advantage
    wind_angles = {
        'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
        'S': 180, 'SW': 225, 'W': 270, 'NW': 315
    }
    wind_angle = wind_angles.get(wind_dir, 0)
    
    # More lenient wind criteria - allow crosswinds
    wind_diff = np.abs(aspect - wind_angle)
    wind_diff = np.minimum(wind_diff, 360 - wind_diff)  # Handle circular nature
    wind_advantage = wind_diff > 30  # Reduced from 45 degrees
    
    # Combine criteria with OR logic for more permissive selection
    buck_potential = high_ground & good_slope & (thermal_cover | wind_advantage)
    
    # Enhanced aggression scaling
    base_iterations = max(1, aggression - 3)
    if aggression > 5:
        buck_potential = binary_dilation(buck_potential, iterations=base_iterations)
    
    # Also add ridge detection for buck beds
    if len(valid_elevation) > 100:
        ridge_threshold = np.percentile(valid_elevation, 75)
        ridges = elevation > ridge_threshold
        buck_potential = buck_potential | (ridges & good_slope)
    
    return extract_points(buck_potential, transform, max_points=20, min_distance_meters=150)

def find_doe_bedding(terrain_data, wind_dir, aggression):
    """Find doe bedding areas - security cover, thermal protection"""
    elevation = terrain_data['elevation']
    slope = terrain_data['slope']
    aspect = terrain_data['aspect']
    transform = terrain_data['transform']
    
    # Enhanced doe bedding criteria
    valid_elevation = elevation[~np.isnan(elevation) & (elevation > 0)]
    if len(valid_elevation) == 0:
        return []
    
    # 1. Security cover - broader elevation range
    mid_elevation = (elevation > np.percentile(valid_elevation, 15)) & \
                   (elevation < np.percentile(valid_elevation, 85))
    
    # 2. Gentle to moderate slopes
    good_slope = (slope > 1) & (slope < 25)
    
    # 3. Enhanced thermal protection using NDVI + terrain
    thermal_protection = np.ones_like(elevation, dtype=bool)
    if terrain_data['ndvi'] is not None:
        ndvi_valid = ~np.isnan(terrain_data['ndvi'])
        # Use NDVI to identify dense vegetation
        dense_vegetation = ndvi_valid & (terrain_data['ndvi'] > 0.35)
        # Also consider moderate vegetation as potential cover
        moderate_vegetation = ndvi_valid & (terrain_data['ndvi'] > 0.25)
        thermal_protection = dense_vegetation | moderate_vegetation
        
        # If no vegetation coverage, use terrain features
        if not np.any(thermal_protection):
            thermal_protection = (elevation > np.percentile(valid_elevation, 30)) & \
                                (elevation < np.percentile(valid_elevation, 70))
    else:
        # Without NDVI, use terrain complexity as cover proxy
        elevation_std = gaussian_filter(elevation, sigma=3)
        terrain_complexity = np.abs(elevation - elevation_std) > np.std(valid_elevation) * 0.3
        thermal_protection = terrain_complexity | mid_elevation
    
    # 4. Wind protection considerations
    wind_angles = {
        'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
        'S': 180, 'SW': 225, 'W': 270, 'NW': 315
    }
    wind_angle = wind_angles.get(wind_dir, 0)
    
    # Prefer leeward aspects but be more flexible
    leeward_angle = (wind_angle + 180) % 360
    wind_diff = np.abs(aspect - leeward_angle)
    wind_diff = np.minimum(wind_diff, 360 - wind_diff)
    wind_protection = wind_diff < 120  # More lenient than before
    
    # Combine criteria with flexible logic
    doe_potential = mid_elevation & good_slope & (thermal_protection | wind_protection)
    
    # Enhanced aggression scaling
    base_iterations = max(1, aggression - 2)
    if aggression > 5:
        doe_potential = binary_dilation(doe_potential, iterations=base_iterations)
    
    return extract_points(doe_potential, transform, max_points=25, min_distance_meters=120)

def filter_points_by_boundary(points, boundary_polygon):
    """Filter points to only include those within the boundary"""
    filtered_points = []
    for lon, lat in points:
        point = Point(lon, lat)
        if boundary_polygon.contains(point):
            filtered_points.append((lon, lat))
    return filtered_points

def find_funnels(terrain_data, aggression):
    """Find natural funnels and travel corridors using improved topographic analysis"""
    elevation = terrain_data['elevation']
    slope = terrain_data['slope']
    transform = terrain_data['transform']
    
    # Remove invalid data
    valid_mask = ~np.isnan(elevation) & (elevation > 0)
    if not np.any(valid_mask):
        return []
    
    # Smooth elevation for better saddle detection
    elevation_smooth = gaussian_filter(elevation, sigma=3)
    
    # Find saddle points using curvature analysis
    gy, gx = np.gradient(elevation_smooth)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    
    # Calculate curvatures
    mean_curvature = (gxx + gyy) / 2
    gaussian_curvature = gxx * gyy - gxy**2
    
    # Saddles have negative Gaussian curvature
    saddles = (gaussian_curvature < -0.0005) & valid_mask
    
    # Find low points between high areas (natural passes)
    # Use morphological operations to find valleys
    from scipy.ndimage import minimum_filter, maximum_filter
    local_min = minimum_filter(elevation_smooth, size=5) == elevation_smooth
    local_max = maximum_filter(elevation_smooth, size=5) == elevation_smooth
    
    # Valleys are areas that are local minimums but not global lows
    valid_elevation = elevation[valid_mask]
    valley_threshold = np.percentile(valid_elevation, 30)
    high_threshold = np.percentile(valid_elevation, 70)
    
    valleys = local_min & (elevation > valley_threshold) & (elevation < high_threshold)
    
    # Gentle slopes for travel corridors
    travel_slopes = (slope > 1) & (slope < 12) & valid_mask
    
    # Combine saddles, valleys, and travel corridors
    funnel_areas = (saddles | valleys) & travel_slopes
    
    # Apply aggression with more conservative dilation
    if aggression > 6:
        funnel_areas = binary_dilation(funnel_areas, iterations=min(aggression-6, 3))
    
    return extract_points(funnel_areas, transform, max_points=12, min_distance_meters=150)

def find_scrape_locations(terrain_data, phase, aggression):
    """Find potential scrape locations based on hunting phase"""
    elevation = terrain_data['elevation']
    slope = terrain_data['slope']
    transform = terrain_data['transform']
    
    if phase == "Pre-Rut":
        # Pre-rut: field edges, transition areas
        moderate_slope = (slope > 1) & (slope < 10)
        edge_areas = moderate_slope
    elif phase == "Rut":
        # Rut: doe bedding areas, funnels
        moderate_slope = (slope > 2) & (slope < 15)
        edge_areas = moderate_slope
    else:  # Early Season
        # Early season: food sources, water
        gentle_slope = (slope > 0.5) & (slope < 8)
        edge_areas = gentle_slope
    
    # Apply aggression
    if aggression > 6:
        edge_areas = binary_dilation(edge_areas, iterations=aggression-6)
    
    return extract_points(edge_areas, transform, max_points=15, min_distance_meters=80)

def extract_points(binary_mask, transform, max_points=20, min_distance_meters=100):
    """Extract coordinate points from binary mask with distance filtering"""
    if not np.any(binary_mask):
        return []
    
    # Find connected components
    labeled, num_features = label(binary_mask)
    
    points = []
    for i in range(1, min(num_features + 1, max_points * 3 + 1)):  # Get more candidates first
        # Find centroid of each component
        component = labeled == i
        if np.sum(component) < 3:  # Skip very small components
            continue
            
        y_coords, x_coords = np.where(component)
        
        # Get centroid
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        # Convert to geographic coordinates
        lon, lat = rasterio.transform.xy(transform, center_y, center_x)
        
        # Check minimum distance from existing points
        too_close = False
        for existing_lon, existing_lat in points:
            # Calculate distance in meters (approximate)
            lat_diff = lat - existing_lat
            lon_diff = lon - existing_lon
            distance_meters = np.sqrt((lat_diff * 111000)**2 + (lon_diff * 111000 * np.cos(np.radians(lat)))**2)
            
            if distance_meters < min_distance_meters:
                too_close = True
                break
        
        if not too_close:
            points.append((lon, lat))
            
        if len(points) >= max_points:
            break
    
    return points

def generate_kml(buck_beds, doe_beds, funnels, scrapes, wind_dir, phase):
    """Generate KML file with hunting locations"""
    
    # Create KML structure
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, "Document")
    
    # Add document info
    name = ET.SubElement(document, "name")
    name.text = f"Digital Deer Scout - {phase} - Wind: {wind_dir}"
    
    # Define styles
    styles = {
        'buck_bed': {'color': 'ff0000ff', 'icon': 'http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png'},
        'doe_bed': {'color': 'ff00ff00', 'icon': 'http://maps.google.com/mapfiles/kml/pushpin/grn-pushpin.png'},
        'funnel': {'color': 'ffffff00', 'icon': 'http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png'},
        'scrape': {'color': 'ffff8000', 'icon': 'http://maps.google.com/mapfiles/kml/pushpin/orange-pushpin.png'}
    }
    
    for style_id, style_props in styles.items():
        style = ET.SubElement(document, "Style", id=style_id)
        icon_style = ET.SubElement(style, "IconStyle")
        color = ET.SubElement(icon_style, "color")
        color.text = style_props['color']
        icon = ET.SubElement(icon_style, "Icon")
        href = ET.SubElement(icon, "href")
        href.text = style_props['icon']
    
    # Add placemarks
    def add_placemark(coords_list, name_prefix, style_id, description=""):
        for i, (lon, lat) in enumerate(coords_list):
            placemark = ET.SubElement(document, "Placemark")
            name_elem = ET.SubElement(placemark, "name")
            name_elem.text = f"{name_prefix} {i+1}"
            
            if description:
                desc_elem = ET.SubElement(placemark, "description")
                desc_elem.text = description
            
            style_url = ET.SubElement(placemark, "styleUrl")
            style_url.text = f"#{style_id}"
            
            point = ET.SubElement(placemark, "Point")
            coordinates = ET.SubElement(point, "coordinates")
            coordinates.text = f"{lon},{lat},0"
    
    # Add all hunting locations
    add_placemark(buck_beds, "Buck Bed", "buck_bed", "Elevated bedding area with good visibility")
    add_placemark(doe_beds, "Doe Bed", "doe_bed", "Security cover bedding area")
    add_placemark(funnels, "Funnel", "funnel", "Natural travel corridor")
    add_placemark(scrapes, "Scrape", "scrape", f"Potential scrape location - {phase}")
    
    # Convert to string
    from xml.dom import minidom
    rough_string = ET.tostring(kml, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# Guard: Require user to upload file
uploaded_file = st.file_uploader("Upload KML or KMZ hunt boundary file", type=["kml", "kmz"])
if uploaded_file:
    gdf = extract_kml(uploaded_file)
    st.success("KML/KMZ file loaded. Proceeding...")
    
    poly = gdf.geometry.iloc[0]
    st.write(f"Boundary bounds: {poly.bounds}")
    area_km2 = poly.area * 111.32 * 111.32
    st.write(f"Boundary area: {area_km2:.2f} km²")

    # DEM Fetch
    st.write("Fetching DEM...")
    try:
        if custom_tiff is not None:
            tiff_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            tiff_file.write(custom_tiff.read())
            tiff_file.close()
            dem_path = tiff_file.name
        else:
            dem_path = fetch_usgs_lidar(poly.bounds)
        st.success("DEM successfully fetched.")
    except Exception as e:
        st.error(f"DEM fetch failed: {e}")
        st.stop()

    # NDVI Fetch
    sentinel_client_id = st.secrets.get("SENTINEL_CLIENT_ID", "")
    sentinel_client_secret = st.secrets.get("SENTINEL_CLIENT_SECRET", "")
    ndvi_path = None
    if sentinel_client_id and sentinel_client_secret:
        st.write("Fetching NDVI...")
        ndvi_path = fetch_sentinel_ndvi(poly.bounds, sentinel_client_id, sentinel_client_secret)
        if ndvi_path:
            st.success("NDVI successfully fetched.")
        else:
            st.warning("NDVI fetch failed or not available.")
    else:
        st.warning("Sentinel credentials missing. Skipping NDVI fetch.")

    # Terrain Analysis and Pin Generation
    st.write("Analyzing terrain and generating hunting locations...")
    
    try:
        # Analyze terrain
        terrain_data = analyze_terrain(dem_path, ndvi_path, poly.bounds)
        
        # Generate hunting locations
        buck_beds = []
        doe_beds = []
        funnels = []
        scrapes = []
        
        if show_buck_beds:
            buck_beds_raw = find_buck_bedding(terrain_data, wind, aggression)
            buck_beds = filter_points_by_boundary(buck_beds_raw, poly)
            st.write(f"Found {len(buck_beds)} buck bedding areas (filtered to boundary)")
            if len(buck_beds) == 0:
                st.info("No buck beds found within boundary. Try increasing aggression level or check if area has sufficient elevation variation.")
        
        if show_doe_beds:
            doe_beds_raw = find_doe_bedding(terrain_data, wind, aggression)
            doe_beds = filter_points_by_boundary(doe_beds_raw, poly)
            st.write(f"Found {len(doe_beds)} doe bedding areas (each represents 3-5 deer family group)")
            if terrain_data['ndvi'] is None:
                st.info("NDVI data not available - using terrain-only analysis for vegetation cover estimation.")
            else:
                ndvi_coverage = np.sum(~np.isnan(terrain_data['ndvi'])) / terrain_data['ndvi'].size
                st.write(f"NDVI coverage: {ndvi_coverage:.1%} of area")
        
        if show_funnels:
            funnels_raw = find_funnels(terrain_data, aggression)
            funnels = filter_points_by_boundary(funnels_raw, poly)
            st.write(f"Found {len(funnels)} funnel locations (natural travel corridors)")
        
        if show_scrapes:
            scrapes_raw = find_scrape_locations(terrain_data, phase, aggression)
            scrapes = filter_points_by_boundary(scrapes_raw, poly)
            st.write(f"Found {len(scrapes)} potential scrape locations")
        
        # Generate KML
        if any([buck_beds, doe_beds, funnels, scrapes]):
            st.write("Generating KML file...")
            kml_content = generate_kml(buck_beds, doe_beds, funnels, scrapes, wind, phase)
            
            # Provide download
            st.download_button(
                label="Download Hunting Locations KML",
                data=kml_content,
                file_name=f"deer_scout_{phase.lower().replace(' ', '_')}_wind_{wind.lower()}.kml",
                mime="application/vnd.google-earth.kml+xml"
            )
            
            st.success("KML file generated successfully!")
            
            # Display summary
            st.subheader("Location Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Buck Beds", len(buck_beds))
            with col2:
                st.metric("Doe Beds", len(doe_beds))
            with col3:
                st.metric("Funnels", len(funnels))
            with col4:
                st.metric("Scrapes", len(scrapes))
        else:
            st.warning("No hunting locations generated. Try adjusting your parameters.")
            
    except Exception as e:
        st.error(f"Terrain analysis failed: {e}")
        st.write("Debug info:", str(e))

else:
    st.warning("Waiting for KML/KMZ upload to proceed.")
    st.stop()