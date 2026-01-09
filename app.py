from pathlib import Path

import geopandas as gpd
import streamlit as st
from dotenv import load_dotenv

from urban_clusters.plot import generate_unique_colors
from urban_clusters.cache import (
    get_hashes,
    get_or_load_hotspot_mask,
    get_or_load_point_clusters,
)
from urban_clusters.scripts.cluster import get_or_load_all_jobs

load_dotenv()

cache_path = Path("./cache")
cache_path.mkdir(exist_ok=True)

with st.sidebar, st.form("spatial_form"):
    bounds_file = st.file_uploader("Upload Bounds File", type=["gpkg", "shp"])

    st.header("Spatial clustering parameters")

    method = st.radio("Method", (1, 2), index=0)
    min_spatial_size = st.number_input(
        "Minimum spatial cluster size (points)",
        min_value=1,
        value=10,
        step=1,
    )
    epsilon = st.number_input("Epsilon (meters)", min_value=0.0, value=100.0, step=1.0)
    spatial_submit = st.form_submit_button("Run Clustering")

if spatial_submit:
    with st.spinner("Loading employment points...", show_time=True):
        all_jobs = get_or_load_all_jobs(cache_path)

    if bounds_file is not None:
        df_bounds = gpd.read_file(bounds_file).to_crs("EPSG:6372")
        df_points = gpd.sjoin(
            all_jobs,
            df_bounds,
            how="inner",
            predicate="within",
        ).drop(columns=["index_right"])
    else:
        st.warning("Please upload a bounds file to proceed.")
        st.stop()

    args_dict = {
        "method": method,
        "min_spatial_size": min_spatial_size,
        "epsilon": epsilon,
    }
    hashes = get_hashes(df_points, args_dict)

    hotspot_cache_path = cache_path / f"{hashes['hotspots']}.npy"
    cluster_cache_path = cache_path / hashes["spatial"] / "points.gpkg"
    cluster_cache_path.parent.mkdir(parents=True, exist_ok=True)

    with st.spinner("Generating hotspot mask...", show_time=True):
        hotspot_mask = get_or_load_hotspot_mask(df_points, hotspot_cache_path)

    with st.spinner("Clustering points...", show_time=True):
        df_points = (
            get_or_load_point_clusters(
                df_points,
                args_dict,
                hotspot_mask,
                cluster_cache_path,
            )
            .to_crs("EPSG:4326")
            .dropna(subset=["spatial_cluster"])
            .assign(
                longitude=lambda df: df.geometry.x,
                latitude=lambda df: df.geometry.y,
                color=lambda df: generate_unique_colors(df["spatial_cluster"])
            )
            .drop(columns=["geometry"])
        )

    st.map(df_points, color="color")

st.write("Hello, Urban Clusters!")
