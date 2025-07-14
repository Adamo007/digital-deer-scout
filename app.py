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
from rasterio.transform import rowcol
from skimage.filters import sobel
import base64
from shapely.geometry import LineString

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
custom_tiff = st.sidebar.file_uploads("Optional: Upload your own GeoTIFF (DEM)", type=["tif", "tiff"])
uploaded_file = st.file_uploader("Upload KML or KMZ hunt boundary file", type=["kml", "kmz"])

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

def fetch_usgs_lidar(bounds, out_path="dem.tif"):
    minx, miny, maxx, maxy = bounds
    if (maxx - minx) > 1 or (maxy - miny) > 1:
        st.error("Bounding box too large for USGS DEM fetch")
        st.stop()
    url = (
        f"https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?bbox={minx},{miny},{maxx},{maxy}"
        f"&bboxSR=4326&imageSR=4326&format=tiff&pixelType=F32&f=image"
    )
    r = requests.get(url)
    if r.status_code == 200 and r.headers.get("Content-Type") == "image/tiff":
        with open(out_path, "wb") as f:
            f.write(r.content)
        if os.path.getsize(out_path) < 2048:
            raiseകException("DEM download resulted in an empty file")
        return out_path
    raise Exception("USGS DEM download failed")

def calculate_slope_aspect(dem_path, geometry):
    with rasterio.open(dem_path) as src:
        out_image, out_transform = mask(src, [geometry], crop=True)
        elevation = out_image[0].astype(float)

        if np.all(elevation == 0) or elevation.max() - elevation.min() < 1:
            raise Exception("DEM data appears to be flat or invalid")

        elevation = gaussian_filter(elevation, sigma=0.75)
        dx, dy = np.gradient(elevation)
        slope = np.sqrt(dx**2 + dy**2)
        aspect = np.arctan2(-dx, dy) * 180 / np.pi
        aspect = np.where(aspect < 0, 360 + aspect, aspect)
        patchcut = sobel(elevation) < 0.08  # Adjusted threshold

        return slope, aspect, out_transform, patchcut, elevation

if uploaded_file:
    gdf = extract_kml(uploaded_file)
    poly = gdf.geometry.iloc[0]
    st.write(f"Boundary bounds: {poly.bounds}")

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

    slope, aspect, transform, patchcut, elevation = calculate_slope_aspect(dem_path, poly)
    st.write(f"Elevation range: {elevation.min()} to {elevation.max()}")
    st.write(f"Aspect range: {aspect.min()} to {aspect.max()}")

    step = max((poly.bounds[2] - poly.bounds[0]) / (aggression * 10), 0.0001)
    candidate_pts = [
        Point(x, y) for x in np.arange(poly.bounds[0], poly.bounds[2], step)
        for y in np.arange(poly.bounds[1], poly.bounds[3], step)
        if poly.contains(Point(x, y))
    ]

    buck_pts, doe_pts, scrape_pts, funnels = [], [], [], []

    elev_range = elevation.max() - elevation.min()
    elev_threshold = (elevation.min() + elev_range * 0.33, elevation.min() + elev_range * 0.66)

    for pt in candidate_pts:
        row, col = rowcol(transform, pt.x, pt.y)
        try:
            s, a, pc = slope[row, col], aspect[row, col], patchcut[row, col]
            elev = elevation[row, col]
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

        if show_buck_beds and 10 < s < 30 and wind_match and elev_threshold[0] < elev < elev_threshold[1]:
            buck_pts.append(pt)
        elif show_doe_beds and s < 5:
            doe_pts.append(pt)

    for b in buck_pts:
        for d in doe_pts:
            if 0.001 < b.distance(d) < 0.005:
                line = LineString([b, d])
                scrape_pts.append(line.interpolate(line.length * 0.5))

    edge_img = sobel(elevation)
    funnel_threshold = np.percentile(edge_img, 95)  # Adjusted threshold
    funnel_indices = np.argwhere(edge_img > funnel_threshold)
    for idx in funnel_indices:
        row, col = idx
        lon, lat = transform * (col, row)
        pt = Point(lon, lat)
        if poly.contains(pt):
            funnels.append(pt)

    pins = []
    if show_buck_beds:
        pins += [{"lon": p.x, "lat": p.y, "type": "Buck Bed", "color": [255, 0, 0]} for p in buck_pts]
    if show_doe_beds:
        pins += [{"lon": p.x, "lat": p.y, "type": "Doe Bed", "color": [255, 255, 0]} for p in doe_pts]
    if show_scrapes:
        pins