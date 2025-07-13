import streamlit as st
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
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
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from rasterio.transform import rowcol
from skimage.filters import sobel
import PIL.Image as Image
from io import BytesIO
import base64

pdk.settings.mapbox_api_key = "pk.eyJ1IjoiYWRhbW8wMDciLCJhIjoiY21kMGNpcms2MTlvaTJscHppNmdtNzRzYyJ9.5g_goxXeEETGRnsTX7y7OA"

st.set_page_config(page_title="Digital Deer Scout AI", layout="wide")
st.title("🧈 Digital Deer Scout – Terrain AI")

# --- Sidebar UI ---
st.sidebar.header("🧠 Scouting Parameters")
wind = st.sidebar.selectbox("Wind Direction", ["NW", "W", "SW", "S", "SE", "E", "NE", "N"])
phase = st.sidebar.selectbox("Target Phase", ["Early Season", "Pre-Rut", "Rut"])
aggression = st.sidebar.slider("Pin Aggression Level", 1, 10, 3)

st.sidebar.markdown("---")
show_buck_beds = st.sidebar.checkbox("Show Buck Bedding", True)
show_doe_beds = st.sidebar.checkbox("Show Doe Bedding", True)
show_scrapes = st.sidebar.checkbox("Show Scrape Locations", True)
show_topo = st.sidebar.checkbox("Show Topographic Overlay", True)
show_debug = st.sidebar.checkbox("Show Candidate Points", False)

uploaded_file = st.file_uploader("Upload your hunt area .KML or .KMZ", type=["kml", "kmz"])

# --- Helper Functions ---
def extract_kml(file) -> gpd.GeoDataFrame:
    if file.name.endswith('.kmz'):
        with ZipFile(file) as zf:
            kml_name = [f for f in zf.namelist() if f.endswith('.kml')][0]
            with zf.open(kml_name) as kml_file:
                gdf = gpd.read_file(kml_file)
    else:
        gdf = gpd.read_file(file)
    return gdf.to_crs("EPSG:4326")

def fetch_usgs_lidar(bounds, out_path="dem.tif"):
    minx, miny, maxx, maxy = bounds
    url = (
        f"https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?bbox={minx},{miny},{maxx},{maxy}"
        f"&bboxSR=4326&imageSR=4326&format=tiff&pixelType=F32&f=image"
    )
    r = requests.get(url)
    if r.status_code == 200 and r.headers['Content-Type'] == 'image/tiff':
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path
    else:
        raise Exception("USGS LiDAR DEM download failed")

def calculate_slope_aspect(dem_path, geometry):
    with rasterio.open(dem_path) as src:
        st.write(f"DEM bounds: {src.bounds}")
        st.write(f"Polygon bounds: {geometry.bounds}")
        out_image, out_transform = mask(src, [geometry], crop=True)
        elevation_data = out_image[0].astype(float)

        elevation_data = gaussian_filter(elevation_data, sigma=0.5)
        x, y = np.gradient(elevation_data)
        slope = np.sqrt(x * x + y * y)
        aspect = np.arctan2(-x, y) * 180 / np.pi
        aspect = np.where(aspect < 0, 360 + aspect, aspect)

        if show_topo:
            fig, ax = plt.subplots()
            ax.imshow(elevation_data, cmap='terrain')
            ax.set_title("Topographic Elevation Map")
            st.pyplot(fig)

        patchcut = sobel(elevation_data) < 0.04

    return slope, aspect, out_transform, patchcut

# [The rest of your code remains unchanged except that fetch_opentopo_dem is replaced by fetch_usgs_lidar]

# Replace call in main logic:
# dem_path = fetch_opentopo_dem(row.geometry.bounds)
# -->
# dem_path = fetch_usgs_lidar(row.geometry.bounds)
