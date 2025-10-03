import geopandas as gpd
import numpy as np
import pandas as pd
from jensen_shannon_centroid import calculate_jsc


def get_distribution_df(
    points: gpd.GeoDataFrame,
    *,
    level: int,
    cluster_col: str | None,
    scian_col: str = "scian",
    normalize: bool = False,
) -> pd.DataFrame | pd.Series:
    """Get the distribution of jobs by industry within each cluster.

    Parameters
    ----------
    points : gpd.GeoDataFrame
        A GeoDataFrame with point geometries, a column indicating the cluster label,
        and a column indicating the industry classification.
    cluster_col : str, optional
        The name of the column in `points` that contains the cluster labels,
        by default "spatial_cluster".
    industry_col : str, optional
        The name of the column in `points` that contains the industry classification,
        by default "subsector".

    Returns
    -------
    pd.DataFrame
        A DataFrame with the distribution of jobs by industry within each cluster.

    """
    points = points.assign(
        industry=lambda df: df[scian_col].astype(str).str[:level].astype(int),
    )

    if cluster_col is None:
        out = points.groupby("industry")["jobs"].sum()
        if normalize:
            out = out / out.sum()
    else:
        out = (
            points.dropna(subset=[cluster_col])
            .assign(**{cluster_col: lambda df: df[cluster_col].astype(int)})
            .groupby([cluster_col, "industry"])["jobs"]
            .sum()
            .reset_index()
            .pivot_table(
                index=cluster_col,
                columns="industry",
                values="jobs",
                fill_value=0,
            )
        )
        if normalize:
            out = out.div(out.sum(axis=1), axis=0)

    return out


def get_location_quotient_df(points: gpd.GeoDataFrame, *, level: int) -> pd.DataFrame:
    num = get_distribution_df(
        points,
        level=level,
        cluster_col="economic_cluster",
        normalize=True,
    )
    den = get_distribution_df(points, level=level, cluster_col=None, normalize=True)

    if not isinstance(num, pd.DataFrame):
        err = "Numerator must be a DataFrame when cluster_col is not None."
        raise TypeError(err)

    if not isinstance(den, pd.Series):
        err = "Denominator must be a Series when cluster_col is None."
        raise TypeError(err)

    return num.div(den, axis=1)


def get_centroid_df(points: gpd.GeoDataFrame, *, level: int = 2) -> pd.DataFrame:
    df_centroids = []
    for economic_cluster, subdf in points.groupby("economic_cluster"):
        if not isinstance(economic_cluster, float):
            err = "Economic cluster must be a float."
            raise TypeError(err)

        temp = gpd.GeoDataFrame(
            subdf.assign(spatial_cluster=lambda df: df["spatial_cluster"].astype(int)),
        )
        subdist = get_distribution_df(
            temp,
            level=level,
            normalize=True,
            cluster_col="spatial_cluster",
        )

        centroid = calculate_jsc(
            subdist.to_numpy()[:, np.newaxis, :],
            T=5000,
        ).flatten()
        centroid_series = pd.Series(centroid, index=subdist.columns)
        df_centroids.append(centroid_series.rename(int(economic_cluster)))

    df_centroids = pd.concat(df_centroids, axis=1).sort_index().fillna(0).transpose()
    return df_centroids.div(df_centroids.sum(axis=1), axis=0)
