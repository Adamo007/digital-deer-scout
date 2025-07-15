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

st.set_page_config(page_title="Digital Deer Scout AI", layout="wide")
st.title("🦌 Digital Deer Scout – Terrain AI")

# Sidebar Controls
st.sidebar.header("🧠 Scouting Parameters")
wind = st.sidebar.selectbox("Wind Direction", ["NW", "W", "SW", "S", "SE", "E", "NE", "N"])
phase = st.sidebar.selectbox("Target Phase", ["Early Season", "Pre-Rut", "Rut"])
aggression = st.sidebar.slider("Pin Aggression Level", 1, 10, 5)
show_buck_beds = st.sidebar.checkbox("Show Buck Bedding", True)
show_doe_beds = st.sidebar.checkbox("Show Doe Bedding", True)
show_scrapes = st.sidebar.checkbox("Show Scrape Locations", True)
show_funnels = st.sidebar.checkbox("Show Funnels", True)
show_topo = st.sidebar.checkbox("Show Topographic Overlay", False)
show_ndvi_heatmap = st.sidebar.checkbox("Show NDVI Heatmap", True)
custom_tiff = st.sidebar.file_uploader("Upload GeoTIFF (DEM)", type=["tif", "tiff"])
custom_ndvi = st.sidebar.file_uploader("Upload NDVI GeoTIFF (optional)", type=["tif", "tiff"])
sentinelhub_token = st.sidebar.text_input("Sentinel Hub Access Token (for NDVI fetch)", help="Generate at https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token")

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
        st.warning("No Sentinel Hub access token provided. Skipping NDVI fetch.")
        return None
    config = SHConfig()
    config.sh_token = token
    config.sh_base_url = "https://services.sentinel-hub.com"
    config.sh_token_url = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
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
    except Exception as e:
        st.error(f"NDVI fetch failed: {e}. Proceeding without NDVI.")
        return None

# Slope, Aspect, TPI, and NDVI Processing
def calculate_slope_aspect_tpi_ndvi(dem_path, ndvi_path, geometry):
    try:
        with rasterio.open(dem_path) as src:
            out_image, out_transform = mask(src, [geometry], crop=True, nodata=np.nan)
            elevation = out_image[0].astype(float)
            if np.all(np.isnan(elevation)) or np.nanmax(elevation) - np.nanmin(elevation) < 1:
                raise Exception("DEM data appears to be flat or invalid")
            elevation = gaussian_filter(elevation, sigma=1.0)
            pixel_size = src.res[0] * 111320  # Approx meters
            dx, dy = np.gradient(elevation, pixel_size)
            slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            slope = np.clip(slope, 0, 90)
            aspect = np.arctan2(-dx, dy) * 180 / np.pi
            aspect = np.where(aspect < 0, 360 + aspect, aspect)
            patchcut = sobel(elevation) < 0.08
            tpi = elevation - gaussian_filter(elevation, sigma=5)

        if ndvi_path:
            with rasterio.open(ndvi_path) as src:
                out_image, _ = mask(src, [geometry], crop=True, nodata=np.nan)
                ndvi = out_image[0].astype(float)
                ndvi = np.where(np.isnan(ndvi), 0, ndvi)
                ndvi = gaussian_filter(ndvi, sigma=1.0)
                if np.all(ndvi == 0):
                    st.warning("NDVI data is all zero. Skipping NDVI.")
                    ndvi = None
        else:
            ndvi = None

        return slope, aspect, out_transform, patchcut, elevation, ndvi, tpi
    except Exception as e:
        st.error(f"Terrain/NDVI processing failed: {e}")
        st.stop()

# Main Logic
uploaded_file = st.file_uploader("Upload KML or KMZ hunt boundary file", type=["kml", "kmz"])
if uploaded_file:
    gdf = extract_kml(uploaded_file)
    poly = gdf.geometry.iloc[0]
    st.write(f"Boundary bounds: {poly.bounds}")
    area_km2 = poly.area * 111.32 * 111.32
    st.write(f"Boundary area: {area_km2:.2f} km²")

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
        else:
            ndvi_path = fetch_sentinel_ndvi(poly.bounds, sentinelhub_token)
        st.success("✅ NDVI successfully fetched or skipped.")
    except Exception as e:
        st.error(f"NDVI fetch failed: {e}. Proceeding without NDVI.")
        ndvi_path = None

    # Process Terrain and NDVI
    slope, aspect, transform, patchcut, elevation, ndvi, tpi = calculate_slope_aspect_tpi_ndvi(dem_path, ndvi_path, poly)
    st.write(f"Elevation range: {np.nanmin(elevation):.2f} to {np.nanmax(elevation):.2f}")
    st.write(f"Aspect range: {np.nanmin(aspect):.2f} to {np.nanmax(aspect):.2f}")
    st.write(f"Slope range: {np.nanmin(slope):.2f} to {np.nanmax(slope):.2f}")
    if ndvi is not None:
        st.write(f"NDVI range: {np.nanmin(ndvi):.2f} to {np.nanmax(ndvi):.2f}")
    else:
        st.write("No NDVI data available.")
    st.write(f"TPI range: {np.nanmin(tpi):.2f} to {np.nanmax(tpi):.2f}")

    # Slope Histogram
    slope_flat = slope.ravel()
    slope_flat = slope_flat[~np.isnan(slope_flat)]
    if len(slope_flat) > 0:
        st.write(f"Slope stats - Mean: {np.mean(slope_flat):.2f}, Median: {np.median(slope_flat):.2f}, "
                 f"10th %: {np.percentile(slope_flat, 10):.2f}, 90th %: {np.percentile(slope_flat, 90):.2f}")

    # Generate Candidate Points
    step = max((poly.bounds[2] - poly.bounds[0]) / (aggression * 30), 0.00002)
    candidate_pts = [
        Point(x + np.random.uniform(-step, step), y + np.random.uniform(-step, step))
        for x in np.arange(poly.bounds[0], poly.bounds[2], step)
        for y in np.arange(poly.bounds[1], poly.bounds[3], step)
        if poly.contains(Point(x, y))
    ]
    st.write(f"Total candidate points: {len(candidate_pts)}")
    st.write(f"Candidate point density: {len(candidate_pts)/area_km2:.2f} points/km²")

    # Pin Generation
    buck_slope_fail, buck_elev_fail, buck_tpi_fail, buck_wind_fail = 0, 0, 0, 0
    doe_slope_fail = 0
    buck_candidates, doe_candidates = 0, 0
    buck_pts, doe_pts, scrape_pts, funnels = [], [], [], []
    elev_range = np.nanmax(elevation) - np.nanmin(elevation)
    elev_threshold = (np.nanmin(elevation), np.nanmax(elevation))
    st.write(f"Elevation threshold for buck beds: {elev_threshold[0]:.2f} to {elev_threshold[1]:.2f}")

    for pt in candidate_pts:
        try:
            row, col = rowcol(transform, pt.x, pt.y)
            s, a, pc = slope[row, col], aspect[row, col], patchcut[row, col]
            elev = elevation[row, col]
            tpi_val = tpi[row, col]
            if np.isnan(s) or np.isnan(a) or np.isnan(elev) or np.isnan(tpi_val):
                continue
        except IndexError:
            continue

        wind_match = (
            (wind == "W" and 180 < a < 360) or
            (wind == "SW" and 135 < a < 315) or
            (wind == "S" and 90 < a < 270) or
            (wind == "SE" and 45 < a < 225) or
            (wind == "E" and 0 < a < 180) or
            (wind == "NE" and (315 < a or a < 135)) or
            (wind == "N" and (270 < a or a < 90)) or
            (wind == "NW" and (225 < a or a < 45))
        )

        # Buck Beds
        if show_buck_beds:
            if not (0.1 < s < 80):
                buck_slope_fail += 1
            elif not (elev_threshold[0] < elev < elev_threshold[1]):
                buck_elev_fail += 1
            elif not (tpi_val > 0.01 or np.abs(tpi_val) < 0.02):
                buck_tpi_fail += 1
            elif not wind_match:
                buck_wind_fail += 1
            else:
                buck_candidates += 1
                buck_pts.append(pt)

        # Doe Beds
        if show_doe_beds:
            if not (s < 15):
                doe_slope_fail += 1
            else:
                doe_candidates += 1
                if ndvi is None or ndvi[row, col] > 0.4:
                    doe_pts.append(pt)

    # Scrapes
    if show_scrapes:
        for b in buck_pts:
            for d in doe_pts:
                if 0.0003 < b.distance(d) < 0.015:
                    line = LineString([b, d])
                    mid_pt = line.interpolate(line.length * 0.5)
                    if ndvi is not None:
                        try:
                            row, col = rowcol(transform, mid_pt.x, pt.y)
                            if 0.1 < ndvi[row, col] < 0.9:
                                scrape_pts.append(mid_pt)
                        except IndexError:
                            continue
                    else:
                        scrape_pts.append(mid_pt)

    # Funnels
    if show_funnels:
        edge_img = sobel(elevation)
        funnel_threshold = np.percentile(edge_img[~np.isnan(edge_img)], 70)
        funnel_indices = np.argwhere(edge_img > funnel_threshold)
        for idx in funnel_indices:
            row, col = idx
            lon, lat = transform * (col, row)
            pt = Point(lon, lat)
            if poly.contains(pt):
                if ndvi is None or (0.1 < ndvi[row, col] < 0.9):
                    funnels.append(pt)

    # Debugging Outputs
    st.write(f"Buck bed candidates: {buck_candidates}")
    st.write(f"Buck bed failures - Slope: {buck_slope_fail}, Elevation: {buck_elev_fail}, TPI: {buck_tpi_fail}, Wind: {buck_wind_fail}")
    st.write(f"Doe bed candidates: {doe_candidates}")
    st.write(f"Doe bed failures - Slope: {doe_slope_fail}")
    st.write(f"Buck bed points: {len(buck_pts)}")
    st.write(f"Doe bed points: {len(doe_pts)}")
    st.write(f"Scrape points: {len(scrape_pts)}")
    st.write(f"Funnel points: {len(funnels)}")

    # Map Visualization
    layers = []
    pins = []
    if show_buck_beds:
        pins += [{"lon": p.x, "lat": p.y, "type": "Buck Bed", "color": [255, 0, 0]} for p in buck_pts]
    if show_doe_beds:
        pins += [{"lon": p.x, "lat": p.y, "type": "Doe Bed", "color": [255, 255, 0]} for p in doe_pts]
    if show_scrapes:
        pins += [{"lon": p.x, "lat": p.y, "type": "Scrape", "color": [200, 0, 200]} for p in scrape_pts]
    if show_funnels:
        pins += [{"lon": p.x, "lat": p.y, "type": "Funnel", "color": [0, 128, 255]} for p in funnels]

    if pins:
        df = pd.DataFrame(pins)
        if df.empty or df[['lat', 'lon']].isna().any().any():
            st.error("Invalid or missing coordinates in pins")
            st.stop()
        st.write(f"Number of pins: {len(pins)}")
        st.write(f"Lat range: {df.lat.min():.6f} to {df.lat.max():.6f}")
        st.write(f"Lon range: {df.lon.min():.6f} to {df.lon.max():.6f}")
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="[lon, lat]",
            get_color="color",
            get_radius=2,
            pickable=True
        ))

    # NDVI Heatmap
    if show_ndvi_heatmap and ndvi is not None and not np.all(ndvi == 0):
        ndvi_data = [
            {"lon": transform * (col, row)[0], "lat": transform * (col, row)[1], "weight": max(min(ndvi[row, col], 1.0), 0.0)}
            for row in range(ndvi.shape[0]) for col in range(ndvi.shape[1]) if not np.isnan(ndvi[row, col]) and ndvi[row, col] != 0
        ]
        if ndvi_data:
            ndvi_df = pd.DataFrame(ndvi_data)
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=ndvi_df,
                get_position="[lon, lat]",
                get_weight="weight",
                opacity=0.3,
                radius_pixels=20
            ))
        else:
            st.warning("No valid NDVI data for heatmap. Check NDVI source.")
    elif show_ndvi_heatmap:
        st.warning("NDVI heatmap disabled: No NDVI data available.")

    if layers:
        try:
            st.pydeck_chart(pdk.Deck(
                map_style="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                initial_view_state=pdk.ViewState(
                    latitude=np.mean([poly.bounds[1], poly.bounds[3]]),
                    longitude=np.mean([poly.bounds[0], poly.bounds[2]]),
                    zoom=15
                ),
                layers=layers
            ))
        except Exception as e:
            st.error(f"Map rendering failed: {e}")
            st.stop()
    else:
        st.warning("No pins or heatmap generated. Check filters or terrain data.")

    # Export KML
    if st.button("Export KML"):
        if pins:
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
        else:
            st.warning("Cannot export KML: No pins generated.")
else:
    st.info("Please upload a KML or KMZ file to begin.")