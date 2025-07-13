import streamlit as st
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
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
from rasterio.transform import rowcol
from skimage.filters import sobel
import base64
from shapely.geometry import LineString

# Mapbox API Key
pdk.settings.mapbox_api_key = "pk.eyJ1IjoiYWRhbW8wMDciLCJhIjoiY21kMGNpcms2MTlvaTJscHppNmdtNzRzYyJ9.5g_goxXeEETGRnsTX7y7OA"

# Streamlit UI
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
show_topo = st.sidebar.checkbox("Show Topographic Overlay", True)
custom_tiff = st.sidebar.file_uploader("Optional: Upload your own GeoTIFF (DEM)", type=["tif", "tiff"])

uploaded_file = st.file_uploader("Upload KML or KMZ hunt boundary file", type=["kml", "kmz"])

# Helper: Load KML
def extract_kml(file) -> gpd.GeoDataFrame:
    if file.name.endswith(".kmz"):
        with ZipFile(file) as zf:
            kml_file = [f for f in zf.namelist() if f.endswith(".kml")][0]
            with zf.open(kml_file) as kmldata:
                gdf = gpd.read_file(kmldata)
    else:
        gdf = gpd.read_file(file)
    return gdf.to_crs("EPSG:4326")

# Helper: Download USGS DEM
def fetch_usgs_lidar(bounds, out_path="dem.tif"):
    minx, miny, maxx, maxy = bounds
    url = (
        f"https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?bbox={minx},{miny},{maxx},{maxy}"
        f"&bboxSR=4326&imageSR=4326&format=tiff&pixelType=F32&f=image"
    )
    r = requests.get(url)
    if r.status_code == 200 and r.headers["Content-Type"] == "image/tiff":
        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path
    else:
        raise Exception("DEM download failed")

# Helper: Analyze Slope/Aspect

def calculate_slope_aspect(dem_path, geometry):
    with rasterio.open(dem_path) as src:
        out_image, out_transform = mask(src, [geometry], crop=True)
        elevation = out_image[0].astype(float)
        elevation = gaussian_filter(elevation, sigma=0.75)

        dx, dy = np.gradient(elevation)
        slope = np.sqrt(dx**2 + dy**2)
        aspect = np.arctan2(-dx, dy) * 180 / np.pi
        aspect = np.where(aspect < 0, 360 + aspect, aspect)

        patchcut = sobel(elevation) < 0.04

        if show_topo:
            fig, ax = plt.subplots()
            ax.imshow(elevation, cmap='terrain')
            ax.set_title("Topographic Elevation Map")
            st.pyplot(fig)

        return slope, aspect, out_transform, patchcut, elevation

# Main Logic
if uploaded_file:
    gdf = extract_kml(uploaded_file)
    poly = gdf.geometry.iloc[0]

    st.write("Fetching DEM...")
    try:
        dem_path = None
        if custom_tiff is not None:
            tiff_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            tiff_file.write(custom_tiff.read())
            tiff_file.close()
            dem_path = tiff_file.name
        else:
            dem_path = fetch_usgs_lidar(poly.bounds)
    except Exception as e:
        st.error(f"DEM fetch failed: {e}")
        st.stop()

    slope, aspect, transform, patchcut, elevation = calculate_slope_aspect(dem_path, poly)

    step = (poly.bounds[2] - poly.bounds[0]) / (aggression * 25)
    candidate_pts = [Point(x, y) for x in np.arange(poly.bounds[0], poly.bounds[2], step)
                     for y in np.arange(poly.bounds[1], poly.bounds[3], step)
                     if poly.contains(Point(x, y))]

    buck_pts, doe_pts, scrape_pts, funnels = [], [], [], []

    for pt in candidate_pts:
        row, col = rowcol(transform, pt.x, pt.y)
        try:
            s, a, pc = slope[row, col], aspect[row, col], patchcut[row, col]
        except:
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

        if show_buck_beds and 6 < s < 35 and wind_match:
            buck_pts.append(pt)
        elif show_doe_beds and s < 4 and pc:
            doe_pts.append(pt)

    for b in buck_pts:
        for d in doe_pts:
            if b.distance(d) < 0.002:
                scrape_pts.append(Point((b.x + d.x)/2, (b.y + d.y)/2))

    # Funnel detection (gradient of elevation > threshold)
    edge_img = sobel(elevation)
    funnel_threshold = np.percentile(edge_img, 97)
    funnel_indices = np.argwhere(edge_img > funnel_threshold)
    for idx in funnel_indices:
        row, col = idx
        lon, lat = transform * (col, row)
        pt = Point(lon, lat)
        if poly.contains(pt):
            funnels.append(pt)

    pins = []
    if show_buck_beds:
        pins += [{"lon": p.x, "lat": p.y, "type": "Buck Bed"} for p in buck_pts]
    if show_doe_beds:
        pins += [{"lon": p.x, "lat": p.y, "type": "Doe Bed"} for p in doe_pts]
    if show_scrapes:
        pins += [{"lon": p.x, "lat": p.y, "type": "Scrape"} for p in scrape_pts]
    pins += [{"lon": p.x, "lat": p.y, "type": "Funnel"} for p in funnels]

    if pins:
        df = pd.DataFrame(pins)
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/satellite-v9",
            initial_view_state=pdk.ViewState(
                latitude=np.mean(df.lat), longitude=np.mean(df.lon), zoom=15),
            layers=[
                pdk.Layer("ScatterplotLayer",
                          data=df,
                          get_position='[lon, lat]',
                          get_color='[200, 30, 0, 160]',
                          get_radius=2,
                          pickable=True)
            ]))

        if st.button("Export KML"):
            kml = Kml()
            for _, r in df.iterrows():
                kml.newpoint(name=r['type'], coords=[(r['lon'], r['lat'])])
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".kml")
            kml.save(tmp.name)
            with open(tmp.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="scout_output.kml">Download KML</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No pins were generated.")
