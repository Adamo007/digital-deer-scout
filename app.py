import streamlit as st
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from simplekml import Kml
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

st.set_page_config(page_title="Digital Deer Scout AI", layout="wide")
st.title("🦌 Digital Deer Scout – Terrain AI")

# Sidebar Controls
st.sidebar.header("🧠 Scouting Parameters")
wind = st.sidebar.selectbox("Wind Direction", ["NW", "W", "SW", "S", "SE", "E", "NE", "N"])
phase = st.sidebar.selectbox("Target Phase", ["Early Season", "Pre-Rut", "Rut"])
aggression = st.sidebar.slider("Pin Aggression Level", 1, 10, 3)
show_buck_beds = st.sidebar.checkbox("Show Buck Bedding", True)
show_doe_beds = st.sidebar.checkbox("Show Doe Bedding", True)
show_scrapes = st.sidebar.checkbox("Show Scrape Locations", True)
show_topo = st.sidebar.checkbox("Show Topographic Overlay", False)
custom_tiff = st.sidebar.file_uploader("Upload GeoTIFF (DEM)", type=["tif", "tiff"])
custom_ndvi = st.sidebar.file_uploader("Upload NDVI GeoTIFF (optional)", type=["tif", "tiff"])
sentinelhub_token = st.sidebar.text_input("Sentinel Hub API Token (optional, for NDVI fetch)")

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

# NDVI Fetch from Sentinel-2
def fetch_sentinel_ndvi(bounds, token, out_path="ndvi.tif"):
    if not token:
        st.error("Sentinel Hub API token required for NDVI fetch")
        st.stop()
    config = SHConfig()
    config.sh_client_id = token
    config.sh_client_secret = token
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
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A)],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(256, 256),
        config=config
    )
    response = request.get_data(save_data=True, save_path=out_path)
    if response and os.path.exists(out_path):
        return out_path
    raise Exception("Sentinel-2 NDVI fetch failed")

# Slope, Aspect, and NDVI Processing
def calculate_slope_aspect_ndvi(dem_path, ndvi_path, geometry):
    with rasterio.open(dem_path) as src:
        out_image, out_transform = mask(src, [geometry], crop=True)
        elevation = out_image[0].astype(float)
        if np.all(elevation == 0) or elevation.max() - elevation.min() < 1:
            raise Exception("DEM data appears to be flat or invalid")
        elevation = gaussian_filter(elevation, sigma=0.75)
        dx, dy = np.gradient(elevation)
        slope = np.degrees(np.sqrt(dx**2 + dy**2))  # Convert to degrees
        aspect = np.arctan2(-dx, dy) * 180 / np.pi
        aspect = np.where(aspect < 0, 360 + aspect, aspect)
        patchcut = sobel(elevation) < 0.08

    if ndvi_path:
        with rasterio.open(ndvi_path) as src:
            out_image, _ = mask(src, [geometry], crop=True)
            ndvi = out_image[0].astype(float)
            ndvi = np.where(np.isnan(ndvi), 0, ndvi)  # Replace NaN with 0
            ndvi = gaussian_filter(ndvi, sigma=0.75)  # Smooth NDVI
    else:
        ndvi = None

    return slope, aspect, out_transform, patchcut, elevation, ndvi

# Main Logic
uploaded_file = st.file_uploader("Upload KML or KMZ hunt boundary file", type=["kml", "kmz"])
if uploaded_file:
    gdf = extract_kml(uploaded_file)
    poly = gdf.geometry.iloc[0]
    st.write(f"Boundary bounds: {poly.bounds}")

    # Fetch or Load DEM
    st.write("Fetching DEM...")
    try:
        if custom_tiff is not None:
            tiff_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            tiff_file.write(custom_tiff.read())
            tiff_file.close()
            dem_path = tiff_file.name
        else:
            dem_path = fetch_usgs_lidar(poly.bounds)
        st.success("✅ DEM successfully fetched.")
    except Exception as e:
        st.error(f"DEM fetch failed: {e}")
        st.stop()

    # Fetch or Load NDVI
    st.write("Fetching NDVI...")
    try:
        if custom_ndvi is not None:
            ndvi_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            ndvi_file.write(custom_ndvi.read())
            ndvi_file.close()
            ndvi_path = ndvi_file.name
        elif sentinelhub_token:
            ndvi_path = fetch_sentinel_ndvi(poly.bounds, sentinelhub_token)
        else:
            st.warning("No NDVI data provided. Skipping vegetation analysis.")
            ndvi_path = None
        st.success("✅ NDVI successfully fetched or skipped.")
    except Exception as e:
        st.error(f"NDVI fetch failed: {e}. Proceeding without NDVI.")
        ndvi_path = None

    # Process Terrain and NDVI
    slope, aspect, transform, patchcut, elevation, ndvi = calculate_slope_aspect_ndvi(dem_path, ndvi_path, poly)
    st.write(f"Elevation range: {elevation.min()} to {elevation.max()}")
    st.write(f"Aspect range: {aspect.min()} to {aspect.max()}")
    if ndvi is not None:
        st.write(f"NDVI range: {ndvi.min()} to {ndvi.max()}")
    st.write(f"Slope range: {slope.min()} to {slope.max()}")  # Added for debugging

    # Generate Candidate Points
    step = max((poly.bounds[2] - poly.bounds[0]) / (aggression * 5), 0.0002)
    candidate_pts = [
        Point(x, y) for x in np.arange(poly.bounds[0], poly.bounds[2], step)
        for y in np.arange(poly.bounds[1], poly.bounds[3], step)
        if poly.contains(Point(x, y))
    ]
    st.write(f"Total candidate points: {len(candidate_pts)}")

    # Pin Generation
    buck_pts, doe_pts, scrape_pts, funnels = [], [], [], []
    elev_range = elevation.max() - elevation.min()
    elev_threshold = (elevation.min() + elev_range * 0.25, elevation.min() + elev_range * 0.75)
    st.write(f"Elevation threshold for buck beds: {elev_threshold}")

    for pt in candidate_pts:
        row, col = rowcol(transform, pt.x, pt.y)
        try:
            s, a, pc = slope[row, col], aspect[row, col], patchcut[row, col]
            elev = elevation[row, col]
            ndvi_val = ndvi[row, col] if ndvi is not None else 0
        except IndexError:
            continue

        wind_match = (
            (wind == "W" and 225 < a < 315) or
            (wind == "SW" and 180 < a < 270) or
            (wind == "S" and 135 < a < 225) or
            (wind == "SE" and 90 < a < 180) or
            (wind == "E" and 45 < a < 135) or
            (wind == "NE" and 0 <= a < 90) or
            (wind == "N" and (315 < a or a < 45)) or
            (wind == "NW" and (270 < a or a < 45))
        )

        # Buck Beds: Slopes, wind-aligned, mid-hill, moderate NDVI
        if show_buck_beds and 5 < s < 35 and wind_match and elev_threshold[0] < elev < elev_threshold[1]:
            if ndvi is None or (0.2 < ndvi_val < 0.6):
                buck_pts.append(pt)
        # Doe Beds: Flat areas, high NDVI
        elif show_doe_beds and s < 8 and (ndvi is None or ndvi_val > 0.4):
            doe_pts.append(pt)

    # Scrapes: Near transitions between buck and doe beds, prefer NDVI edges
    for b in buck_pts:
        for d in doe_pts:
            if 0.0005 < b.distance(d) < 0.01:
                line = LineString([b, d])
                mid_pt = line.interpolate(line.length * 0.5)
                if ndvi is not None:
                    row, col = rowcol(transform, mid_pt.x, mid_pt.y)
                    try:
                        ndvi_val = ndvi[row, col]
                        if 0.3 < ndvi_val < 0.7:
                            scrape_pts.append(mid_pt)
                    except IndexError:
                        continue
                else:
                    scrape_pts.append(mid_pt)

    # Funnels: Topographic constrictions, prefer NDVI transitions
    edge_img = sobel(elevation)
    funnel_threshold = np.percentile(edge_img, 90)
    funnel_indices = np.argwhere(edge_img > funnel_threshold)
    for idx in funnel_indices:
        row, col = idx
        lon, lat = transform * (col, row)
        pt = Point(lon, lat)
        if poly.contains(pt) and (ndvi is None or 0.3 < ndvi[row, col] < 0.7):
            funnels.append(pt)

    # Create Pins
    pins = []
    if show_buck_beds:
        pins += [{"lon": p.x, "lat": p.y, "type": "Buck Bed", "color": [255, 0, 0]} for p in buck_pts]
    if show_doe_beds:
        pins += [{"lon": p.x, "lat": p.y, "type": "Doe Bed", "color": [255, 255, 0]} for p in doe_pts]
    if show_scrapes:
        pins += [{"lon": p.x, "lat": p.y, "type": "Scrape", "color": [200, 0, 200]} for p in scrape_pts]
    pins += [{"lon": p.x, "lat": p.y, "type": "Funnel", "color": [0, 128, 255]} for p in funnels]

    # Debugging Outputs
    st.write(f"Buck bed points: {len(buck_pts)}")
    st.write(f"Doe bed points: {len(doe_pts)}")
    st.write(f"Scrape points: {len(scrape_pts)}")
    st.write(f"Funnel points: {len(funnels)}")

    # Map Visualization
    if pins:
        df = pd.DataFrame(pins)
        if df.empty or df[['lat', 'lon']].isna().any().any():
            st.error("Invalid or missing coordinates in pins")
            st.stop()
        st.write(f"Number of pins: {len(pins)}")
        try:
            st.pydeck_chart(pdk.Deck(
                map_style="https://tile.opentopomap.org/{z}/{x}/{y}.png",
                initial_view_state=pdk.ViewState(
                    latitude=np.mean(df.lat), longitude=np.mean(df.lon), zoom=15),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=df,
                        get_position="[lon, lat]",
                        get_color="color",
                        get_radius=2,
                        pickable=True
                    )
                ]
            ))
        except Exception as e:
            st.error(f"Map rendering failed: {e}")
            st.stop()
    else:
        st.warning("No pins were generated. Try adjusting aggression, slope, or NDVI thresholds.")

    # Export KML
    if st.button("Export KML") and pins:
        try:
            gdf_pins = gpd.GeoDataFrame(
                df, geometry=[Point(row['lon'], row['lat']) for _, row in df.iterrows()], crs="EPSG:4326"
            )
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".kml")
            gdf_pins.to_file(tmp.name, driver="KML")
            with open(tmp.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="scout_output.kml">Download KML</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"KML export failed: {e}")
    elif st.button("Export KML") and not pins:
        st.warning("Cannot export KML: No pins generated.")
else:
    st.info("Please upload a KML or KMZ file to begin.")