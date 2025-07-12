import streamlit as st
import geopandas as gpd
from shapely.geometry import Point, Polygon
from simplekml import Kml
import tempfile
import numpy as np
import requests
import rasterio
import rasterio.io
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as Rsp
from rasterio.crs import CRS
from rasterio.mask import mask
from pyproj import Transformer

st.set_page_config(page_title="Cloud-Based Digital Deer Scout", layout="centered")
st.title("☁️ Digital Deer Scout – Cloud Terrain AI")

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
    from zipfile import ZipFile
    if file.name.endswith('.kmz'):
        with ZipFile(file) as zf:
            kml_name = [f for f in zf.namelist() if f.endswith('.kml')][0]
            with zf.open(kml_name) as kml_file:
                gdf = gpd.read_file(kml_file)
    else:
        gdf = gpd.read_file(file)
    return gdf.to_crs("EPSG:4326")

def fetch_usgs_dem(bounds):
    minx, miny, maxx, maxy = bounds
    url = (
        f"https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?"
        f"bbox={minx},{miny},{maxx},{maxy}&bboxSR=4326&size=512,512&imageSR=4326&format=tiff&f=image"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception("DEM download failed")
    return resp.content

def calculate_slope_aspect_from_bytes(tiff_bytes, geometry):
    with rasterio.io.MemoryFile(tiff_bytes) as memfile:
        with memfile.open() as src:
            out_image, out_transform = mask(src, [geometry], crop=True)
            elevation = out_image[0].astype(float)

            x, y = np.gradient(elevation)
            slope = np.sqrt(x*x + y*y)
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

def generate_terrain_pins(geometry, wind, level, dem_bytes):
    slope, aspect = calculate_slope_aspect_from_bytes(dem_bytes, geometry)
    buck_pins, doe_pins, scrape_pins = [], [], []

    bounds = geometry.bounds
    minx, miny, maxx, maxy = bounds
    step = (maxx - minx) / (level * 4)

    for i in range(level * 4):
        for j in range(level * 4):
            px = minx + step * i
            py = miny + step * j
            p = Point(px, py)
            if geometry.contains(p):
                sl = slope[j % slope.shape[0]][i % slope.shape[1]]
                asp = aspect[j % aspect.shape[0]][i % aspect.shape[1]]

                if show_buck_beds and 5 < sl < 30 and aspect_matches_wind(asp, wind):
                    buck_pins.append(p)
                elif show_doe_beds and sl < 5:
                    doe_pins.append(p)

    for b in buck_pins:
        for d in doe_pins:
            if b.distance(d) < 0.002:
                mid = Point((b.x + d.x)/2, (b.y + d.y)/2)
                scrape_pins.append(mid)

    return buck_pins[:level], doe_pins[:level], scrape_pins[:level]

# --- Main Logic ---
if uploaded_file:
    gdf = extract_kml(uploaded_file)
    st.success("✅ Hunt boundary loaded.")

    if st.button("🎯 Generate AI Pins"):        
        kml = Kml()

        for _, row in gdf.iterrows():
            if isinstance(row.geometry, Polygon):
                dem_bytes = fetch_usgs_dem(row.geometry.bounds)
                buck_pins, doe_pins, scrape_pins = generate_terrain_pins(row.geometry, wind, aggression, dem_bytes)

                if show_buck_beds:
                    for pt in buck_pins:
                        kml.newpoint(name="Buck Bed", coords=[(pt.x, pt.y)])

                if show_doe_beds:
                    for pt in doe_pins:
                        kml.newpoint(name="Doe Bed", coords=[(pt.x, pt.y)])

                if show_scrapes:
                    for pt in scrape_pins:
                        kml.newpoint(name="Scrape", coords=[(pt.x, pt.y)])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
            kml.save(tmp.name)
            st.download_button("📥 Download AI Pins (KML)", data=open(tmp.name, 'rb'), file_name="terrain_scouting.kml")

        st.success("📌 Terrain-based pins generated and export-ready.")
