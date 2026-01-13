from pathlib import Path

import folium
import geopandas as gpd
import shapely
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

from urban_clusters.cache import (
    get_or_load_filtered_points,
    get_or_load_hotspot_mask,
    get_or_load_point_clusters,
)
from urban_clusters.plot import generate_unique_colors
from urban_clusters.spatial import generate_hulls

load_dotenv()

st.set_page_config(  # Alternate names: setup_page, page, layout
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
)

cache_path = Path("./cache")
cache_path.mkdir(exist_ok=True)

with st.sidebar, st.form("spatial_form"):
    bounds_file = st.file_uploader("Upload Bounds File", type=["gpkg", "shp"])

    st.header("Spatial clustering parameters")

    method = st.radio("Method", (1, 2), index=1)
    min_spatial_size = st.number_input(
        "Minimum spatial cluster size (points)",
        min_value=1,
        value=50,
        step=1,
    )
    epsilon = st.number_input("Epsilon (meters)", min_value=0.0, value=400.0, step=1.0)
    spatial_submit = st.form_submit_button("Run Clustering")

args_dict = {
    "method": method,
    "min_spatial_size": min_spatial_size,
    "epsilon": epsilon,
}

if spatial_submit:
    if bounds_file is not None:
        df_bounds = gpd.read_file(bounds_file).to_crs("EPSG:6372")
        with st.spinner("Filtering points...", show_time=True):
            df_points = get_or_load_filtered_points(
                df_bounds,
                cache_path=cache_path,
            )
    else:
        st.warning("Please upload a bounds file to proceed.")
        st.stop()

    with st.spinner("Generating hotspot mask...", show_time=True):
        hotspot_mask = get_or_load_hotspot_mask(df_points, cache_path=cache_path)

    with st.spinner("Clustering points...", show_time=True):
        df_points = get_or_load_point_clusters(
            df_points,
            args_dict,
            hotspot_mask,
            cache_path=cache_path,
        )

        df_hulls = generate_hulls(
            df_points,
            cluster_col="spatial_cluster",
        ).assign(color=lambda df: generate_unique_colors(df["spatial_cluster"]))

    hull_centroids = df_points["geometry"].to_crs("EPSG:4326").tolist()
    multipoint = shapely.geometry.MultiPoint(hull_centroids)  # pyright: ignore[reportArgumentType]
    loc = multipoint.representative_point().coords[0][::-1]
    if len(loc) != 2:
        err = "Could not determine map center from points."
        raise ValueError(err)

    m = folium.Map(
        location=loc,
        zoom_start=12,
        width=1200,
        height=800,
        tiles="cartodb positron",
    )
    hulls_geo = folium.GeoJson(
        df_hulls.to_crs("EPSG:4326").to_json(),
        style_function=lambda x: {
            "fillColor": x["properties"]["color"],
            "color": x["properties"]["color"],
            "weight": 2,
            "fillOpacity": 0.4,
        },
    )
    hulls_geo.add_to(m)

    st_folium(
        m,
        use_container_width=True,
        returned_objects=[],
    )

    cl, cr = st.columns(2)
    with cl:
        clustered_download = st.download_button(
            label="Download clustered points",
            data=df_points.to_json(drop_id=False),
            file_name="clustered_points.geojson",
            mime="application/json",
            on_click="ignore",
        )
    with cr:
        hull_download = st.download_button(
            label="Download cluster hulls",
            data=df_hulls.to_json(drop_id=False),
            file_name="cluster_hulls.geojson",
            mime="application/json",
            on_click="ignore",
        )
