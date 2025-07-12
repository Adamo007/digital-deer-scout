import streamlit as st
import geopandas as gpd
from shapely.geometry import Point, Polygon
from simplekml import Kml
import tempfile
import rasterio
import numpy as np
import os
from rasterio.mask import mask
from zipfile import ZipFile
import requests
from scipy.ndimage import gaussian_filter

st.set_page_config(page_title="Digital Deer Scout AI", layout="centered")
st.title("🦌 Digital Deer Scout – Terrain AI")

# --- Sidebar UI ---
st.sidebar.header("🧠 Scouting Parameters")
wind = st.sidebar.selectbox("Wind Direction", ["NW", "W", "SW", "S", "SE", "E", "NE", "N"])
phase = st.sidebar.selectbox("Target Phase", ["Early Season", "Pre-Rut", "Rut"])
aggression = st.sidebar.slider("Pin Aggression Level", 1, 10, 3)

st.sidebar.markdown("---")
show_buck_beds = st.sidebar.checkbox("Show Buck Bedding", True)
show_doe_beds = st.sidebar.checkbox("Show Doe Bedding", True)
show_scrapes = st.sidebar.checkbox("Show Scrape Locations", True)

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

        elevation_data = gaussian_filter(elevation_data, sigma=1)  # smooth out jagged noise
        x, y = np.gradient(elevation_data)
        slope = np.sqrt(x * x + y * y)
        aspect = np.arctan2(-x, y) * 180 / np.pi
        aspect = np.where(aspect < 0, 360 + aspect, aspect)

    return slope, aspect, out_transform

def aspect_matches_wind(aspect, wind):
    wind_aspects = {
        "N": 180, "NE": 225, "E": 270, "SE": 315,
        "S": 0,   "SW": 45,  "W": 90,  "NW": 135
    }
    expected = wind_aspects.get(wind, 180)
    return abs(aspect - expected) < 45

def sample_features(slope, aspect, transform, geometry, level):
    buck_pins, doe_pins = [], []
    count = 0
    max_points = 100 * level
    rows, cols = slope.shape

    for r in range(rows):
        for c in range(cols):
            if count > max_points:
                break
            sl = slope[r][c]
            asp = aspect[r][c]
            x, y = rasterio.transform.xy(transform, r, c)
            pt = Point(x, y)

            if not geometry.contains(pt):
                continue

            if show_buck_beds and 5 < sl < 30 and aspect_matches_wind(asp, wind):
                buck_pins.append(pt)
                count += 1
            elif show_doe_beds and sl < 5:
                doe_pins.append(pt)
                count += 1

    return buck_pins[:level*3], doe_pins[:level*5]

def generate_terrain_pins(geometry, wind, level, dem_path):
    slope, aspect, transform = calculate_slope_aspect(dem_path, geometry)
    buck_pins, doe_pins = sample_features(slope, aspect, transform, geometry, level)

    scrape_pins = []
    if show_scrapes:
        for b in buck_pins:
            for d in doe_pins:
                if b.distance(d) < 0.003:
                    mid = Point((b.x + d.x)/2, (b.y + d.y)/2)
                    scrape_pins.append(mid)

    return buck_pins, doe_pins, scrape_pins[:level * 3]

# --- Main Logic ---
if uploaded_file:
    gdf = extract_kml(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_dem:
        try:
            bounds = gdf.total_bounds
            fetch_opentopo_dem(bounds, tmp_dem.name)
            dem_path = tmp_dem.name
            st.success("✅ DEM fetched from OpenTopography")
        except Exception as e:
            st.error(f"❌ DEM fetch failed: {e}")
            st.stop()

    if st.button("🎯 Generate AI Pins"):
        kml = Kml()
        total_buck, total_doe, total_scrape = 0, 0, 0

        for _, row in gdf.iterrows():
            if isinstance(row.geometry, Polygon):
                buck_pins, doe_pins, scrape_pins = generate_terrain_pins(row.geometry, wind, aggression, dem_path)

                for pt in buck_pins:
                    kml.newpoint(name="Buck Bed", coords=[(pt.x, pt.y)])
                    total_buck += 1

                for pt in doe_pins:
                    kml.newpoint(name="Doe Bed", coords=[(pt.x, pt.y)])
                    total_doe += 1

                for pt in scrape_pins:
                    kml.newpoint(name="Scrape", coords=[(pt.x, pt.y)])
                    total_scrape += 1

        with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
            kml.save(tmp.name)
            st.download_button("📥 Download AI Pins (KML)", data=open(tmp.name, 'rb'), file_name="terrain_scouting.kml")

        st.success(f"📌 Generated {total_buck} buck beds, {total_doe} doe beds, and {total_scrape} scrapes.")
