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
from rasterio.transform import rowcol

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

        x, y = np.gradient(elevation_data)
        slope = np.sqrt(x * x + y * y)
        aspect = np.arctan2(-x, y) * 180 / np.pi
        aspect = np.where(aspect < 0, 360 + aspect, aspect)

    return slope, aspect

def aspect_matches_wind(aspect, wind):
    wind_aspects = {
        "N": 180, "NE": 225, "E": 270, "SE": 315,
        "S": 0,   "SW": 45,  "W": 90,  "NW": 135
    }
    expected = wind_aspects.get(wind, 180)
    return abs(aspect - expected) < 45

def generate_terrain_pins(geometry, wind, level, dem_path):
    buck_pins, doe_pins, scrape_pins = [], [], []

    with rasterio.open(dem_path) as src:
        out_image, out_transform = mask(src, [geometry], crop=True)
        elevation_data = out_image[0].astype(float)
        slope = np.hypot(*np.gradient(elevation_data))
        aspect = np.arctan2(-np.gradient(elevation_data)[1], np.gradient(elevation_data)[0]) * 180 / np.pi
        aspect = np.where(aspect < 0, 360 + aspect, aspect)

        bounds = geometry.bounds
        minx, miny, maxx, maxy = bounds
        step = (maxx - minx) / (level * 8)

        for i in range(level * 8):
            for j in range(level * 8):
                px = minx + step * i
                py = miny + step * j
                point = Point(px, py)

                if geometry.contains(point):
                    try:
                        row, col = rowcol(out_transform, px, py)
                        sl = slope[row, col]
                        asp = aspect[row, col]

                        if show_buck_beds and 5 < sl < 30 and aspect_matches_wind(asp, wind):
                            buck_pins.append(point)
                        elif show_doe_beds and sl < 5:
                            doe_pins.append(point)
                    except IndexError:
                        continue

    for b in buck_pins:
        for d in doe_pins:
            if b.distance(d) < 0.002:
                mid = Point((b.x + d.x)/2, (b.y + d.y)/2)
                scrape_pins.append(mid)

    return buck_pins[:level * 2], doe_pins[:level * 2], scrape_pins[:level * 2]

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
            st.download_button("📅 Download AI Pins (KML)", data=open(tmp.name, 'rb'), file_name="terrain_scouting.kml")

        st.success(f"📌 Generated {total_buck} buck beds, {total_doe} doe beds, and {total_scrape} scrapes.")