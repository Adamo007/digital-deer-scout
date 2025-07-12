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

st.set_page_config(page_title="Digital Deer Scout AI", layout="wide")
st.title("🪼 Digital Deer Scout – Terrain AI")

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

def fetch_opentopo_dem(bounds, out_path="dem.tif"):
    minx, miny, maxx, maxy = bounds
    api_key = st.secrets["OPENTOPO_KEY"] if "OPENTOPO_KEY" in st.secrets else "demotoken"
    url = (
        f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1"
        f"&south={miny}&north={maxy}&west={minx}&east={maxx}&outputFormat=GTiff"
        f"&API_Key={api_key}"
    )
    r = requests.get(url)
    if r.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path
    else:
        raise Exception("DEM download failed")

def calculate_slope_aspect(dem_path, geometry):
    with rasterio.open(dem_path) as src:
        out_image, out_transform = mask(src, [geometry], crop=True)
        elevation_data = out_image[0].astype(float)

        elevation_data = gaussian_filter(elevation_data, sigma=0.6)
        x, y = np.gradient(elevation_data)
        slope = np.sqrt(x * x + y * y)
        aspect = np.arctan2(-x, y) * 180 / np.pi
        aspect = np.where(aspect < 0, 360 + aspect, aspect)

        if show_topo:
            fig, ax = plt.subplots()
            ax.imshow(elevation_data, cmap='terrain')
            ax.set_title("Topographic Elevation Map")
            st.pyplot(fig)

        patchcut = sobel(elevation_data) < 0.05

    return slope, aspect, out_transform, patchcut

def aspect_matches_wind(aspect, wind):
    wind_aspects = {
        "N": 180, "NE": 225, "E": 270, "SE": 315,
        "S": 0,   "SW": 45,  "W": 90,  "NW": 135
    }
    expected = wind_aspects.get(wind, 180)
    return abs(aspect - expected) < 45

def sample_features(slope, aspect, transform, geometry, patchcut, level):
    buck_pins, doe_pins = [], []
    slope_thresh = (3, 32)
    flat_mask = slope < 3
    buck_mask = (slope > slope_thresh[0]) & (slope < slope_thresh[1])

    rows, cols = slope.shape
    max_pins = level * 10

    candidate_indices = [
        (r, c)
        for r in range(5, rows - 5)
        for c in range(5, cols - 5)
        if geometry.contains(Point(*rasterio.transform.xy(transform, r, c, offset='center')))
    ]

    for r, c in candidate_indices:
        x, y = rasterio.transform.xy(transform, r, c, offset='center')
        pt = Point(x, y)
        sl = slope[r][c]
        asp = aspect[r][c]

        if show_buck_beds and buck_mask[r, c] and aspect_matches_wind(asp, wind) and patchcut[r, c]:
            buck_pins.append(pt)

        elif show_doe_beds and flat_mask[r, c] and not patchcut[r, c]:
            doe_pins.append(pt)

        if len(buck_pins) >= max_pins and len(doe_pins) >= max_pins:
            break

    return buck_pins[:max_pins], doe_pins[:max_pins]

def predict_funnels(pins):
    if len(pins) < 2:
        return []
    coords = np.array([[p.x, p.y] for p in pins])
    clusters = DBSCAN(eps=0.0015, min_samples=2).fit(coords)
    funnels = []
    for cluster_id in set(clusters.labels_):
        if cluster_id == -1:
            continue
        cluster_points = coords[clusters.labels_ == cluster_id]
        if len(cluster_points) >= 2:
            line = LineString(cluster_points)
            funnels.append(line)
    return funnels

def generate_terrain_pins(geometry, wind, level, dem_path):
    slope, aspect, transform, patchcut = calculate_slope_aspect(dem_path, geometry)
    buck_pins, doe_pins = sample_features(slope, aspect, transform, geometry, patchcut, level)

    scrape_pins = []
    if show_scrapes:
        for b in buck_pins:
            for d in doe_pins:
                if b.distance(d) < 0.003:
                    mid = Point((b.x + d.x)/2, (b.y + d.y)/2)
                    scrape_pins.append(mid)
        scrape_pins = list(unary_union(scrape_pins).geoms) if scrape_pins else []

    return buck_pins, doe_pins, scrape_pins, predict_funnels(buck_pins + doe_pins)
