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

st.set_page_config(page_title="Digital Deer Scout AI", layout="wide")
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

        elevation_data = gaussian_filter(elevation_data, sigma=1)
        x, y = np.gradient(elevation_data)
        slope = np.sqrt(x * x + y * y)
        aspect = np.arctan2(-x, y) * 180 / np.pi
        aspect = np.where(aspect < 0, 360 + aspect, aspect)

        if show_topo:
            fig, ax = plt.subplots()
            ax.imshow(elevation_data, cmap='terrain')
            ax.set_title("Topographic Elevation Map")
            st.pyplot(fig)

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
    slope_thresh = (5, 30)
    flat_mask = slope < 3
    buck_mask = (slope > slope_thresh[0]) & (slope < slope_thresh[1])

    rows, cols = slope.shape
    max_pins = level * 10
    rng = np.random.default_rng(42)

    candidate_indices = [
        (r, c)
        for r in range(5, rows - 5)
        for c in range(5, cols - 5)
        if geometry.contains(Point(*rasterio.transform.xy(transform, r, c, offset='center')))
    ]
    rng.shuffle(candidate_indices)

    for r, c in candidate_indices:
        x, y = rasterio.transform.xy(transform, r, c, offset='center')
        pt = Point(x, y)
        sl = slope[r][c]
        asp = aspect[r][c]

        if show_buck_beds and buck_mask[r, c] and aspect_matches_wind(asp, wind):
            below = slope[min(rows - 1, r + 3)][c]
            if below < sl:
                buck_pins.append(pt)

        elif show_doe_beds and flat_mask[r, c]:
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
    slope, aspect, transform = calculate_slope_aspect(dem_path, geometry)
    buck_pins, doe_pins = sample_features(slope, aspect, transform, geometry, level)

    scrape_pins = []
    if show_scrapes:
        for b in buck_pins:
            for d in doe_pins:
                if b.distance(d) < 0.003:
                    mid = Point((b.x + d.x)/2, (b.y + d.y)/2)
                    scrape_pins.append(mid)
        scrape_pins = list(unary_union(scrape_pins).geoms) if scrape_pins else []

    return buck_pins, doe_pins, scrape_pins, predict_funnels(buck_pins + doe_pins)

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
        buck_coords, doe_coords, scrape_coords, funnel_coords = [], [], [], []

        for _, row in gdf.iterrows():
            if isinstance(row.geometry, Polygon):
                buck_pins, doe_pins, scrape_pins, funnels = generate_terrain_pins(row.geometry, wind, aggression, dem_path)

                for pt in buck_pins:
                    kml.newpoint(name="Buck Bed", coords=[(pt.x, pt.y)])
                    total_buck += 1
                    buck_coords.append({"lat": pt.y, "lon": pt.x})

                for pt in doe_pins:
                    kml.newpoint(name="Doe Bed", coords=[(pt.x, pt.y)])
                    total_doe += 1
                    doe_coords.append({"lat": pt.y, "lon": pt.x})

                for pt in scrape_pins:
                    kml.newpoint(name="Scrape", coords=[(pt.x, pt.y)])
                    total_scrape += 1
                    scrape_coords.append({"lat": pt.y, "lon": pt.x})

                for line in funnels:
                    coords = list(line.coords)
                    for i in range(len(coords)-1):
                        ls = kml.newlinestring(name="Funnel", coords=[coords[i], coords[i+1]])
                        funnel_coords.append({"lat": coords[i][1], "lon": coords[i][0], "type": "Funnel", "color": [0, 255, 255]})

        with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
            kml.save(tmp.name)
            st.download_button("📅 Download AI Pins (KML)", data=open(tmp.name, 'rb'), file_name="terrain_scouting.kml")

        st.success(f"📌 Generated {total_buck} buck beds, {total_doe} doe beds, and {total_scrape} scrapes.")

        # --- Interactive Visualization ---
        st.subheader("📜 Interactive Pin Map Preview")
        all_coords = []
        for p in buck_coords:
            p["type"] = "Buck Bed"
            p["color"] = [255, 0, 0]
            all_coords.append(p)
        for p in doe_coords:
            p["type"] = "Doe Bed"
            p["color"] = [0, 255, 0]
            all_coords.append(p)
        for p in scrape_coords:
            p["type"] = "Scrape"
            p["color"] = [255, 255, 0]
            all_coords.append(p)
        for p in funnel_coords:
            all_coords.append(p)

        if all_coords:
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/satellite-streets-v11",
                initial_view_state=pdk.ViewState(
                    latitude=np.mean([p["lat"] for p in all_coords]),
                    longitude=np.mean([p["lon"] for p in all_coords]),
                    zoom=14,
                    pitch=30,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=[p for p in all_coords if p["type"] != "Funnel"],
                        get_position='[lon, lat]',
                        get_color='color',
                        get_radius=8,
                        pickable=True,
                    ),
                    pdk.Layer(
                        "PathLayer",
                        data=[{
                            "path": [[p["lon"], p["lat"]] for p in funnel_coords],
                            "color": [0, 255, 255],
                        }],
                        get_path="path",
                        get_color="color",
                        width_scale=5,
                        width_min_pixels=2,
                    )
                ],
                tooltip={"text": "{type}"},
                map_provider='mapbox',
            ))
