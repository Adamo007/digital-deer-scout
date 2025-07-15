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
st.sidebar.header("🧐 Scouting Parameters")
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

# Guard: Require user to upload file
uploaded_file = st.file_uploader("Upload KML or KMZ hunt boundary file", type=["kml", "kmz"])
if uploaded_file:
    gdf = extract_kml(uploaded_file)
    st.success("✅ KML/KMZ file loaded. Proceeding...")
    st.write(gdf)

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
        st.success("✅ DEM successfully fetched.")
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
            st.success("✅ NDVI successfully fetched.")
        else:
            st.warning("⚠️ NDVI fetch failed or not available.")
    else:
        st.warning("Sentinel credentials missing. Skipping NDVI fetch.")

    # TODO: Add back in terrain processing and pin placement logic here.
# Calculate slope, aspect, TPI
def calculate_slope_aspect_tpi(dem_path, geometry):
    with rasterio.open(dem_path) as src:
        out_image, out_transform = mask(src, [geometry], crop=True, nodata=np.nan)
        elevation = out_image[0].astype(float)
        elevation = gaussian_filter(elevation, sigma=2.0)
        pixel_size = src.res[0] * 111320
        dx, dy = np.gradient(elevation, pixel_size)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        aspect = np.arctan2(-dx, dy) * 180 / np.pi
        aspect = np.where(aspect < 0, 360 + aspect, aspect)
        tpi = elevation - gaussian_filter(elevation, sigma=5)
        return slope, aspect, tpi, elevation, out_transform

# Process terrain
slope, aspect, tpi, elevation, transform = calculate_slope_aspect_tpi(dem_path, poly)
st.success("✅ Terrain processed.")

# Generate candidate pins
step = 0.0002
x_coords = np.arange(poly.bounds[0], poly.bounds[2], step)
y_coords = np.arange(poly.bounds[1], poly.bounds[3], step)
candidate_pts = [Point(x, y) for x in x_coords for y in y_coords if poly.contains(Point(x, y))]

# Classify pins (simple version)
buck_pts, doe_pts = [], []
for pt in candidate_pts:
    try:
        row, col = rowcol(transform, pt.x, pt.y)
        s = slope[row, col]
        a = aspect[row, col]
        t = tpi[row, col]
      if show_buck_beds:
    if not (3 < s < 40):  # moderate slope
        buck_slope_fail += 1
    elif aggression < 8 and not (elev_threshold[0] + 0.1 * elev_range < elev < elev_threshold[1] - 0.1 * elev_range):
        buck_elev_fail += 1
    elif not (-0.1 < tpi_val < 1.0):  # relaxed topographic prominence
        buck_tpi_fail += 1
    elif not wind_match:
        buck_wind_fail += 1
    elif ndvi is not None and ndvi[row, col] < 0.05:
        pass  # skip low vegetation
    else:
        buck_candidates += 1
        buck_pts.append(pt)

    except:
        continue

# Assemble pins
pins = []
pins += [{"lon": pt.x, "lat": pt.y, "type": "Buck Bed", "color": [255, 0, 0]} for pt in buck_pts]
pins += [{"lon": pt.x, "lat": pt.y, "type": "Doe Bed", "color": [255, 255, 0]} for pt in doe_pts]

if not pins:
    st.warning("⚠️ No pins were generated. Adjust your parameters or check the DEM.")
else:
    df = pd.DataFrame(pins)
    st.success(f"✅ {len(df)} pins generated.")
    st.dataframe(df)

    # KML Export
    if st.button("Export KML"):
        try:
            gdf_pins = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".kml")
            gdf_pins.to_file(tmp.name, driver="KML")
            with open(tmp.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="scout_output.kml">Download KML</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"KML export failed: {e}")

    st.write("🔧 Terrain processing and pin placement logic would be executed here...")
else:
    st.warning("⏳ Waiting for KML/KMZ upload to proceed.")
    st.stop()
