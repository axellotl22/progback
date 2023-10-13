"""
n_d_basic_kmeans_service.py
-----------------------
Service for performing N-Dimensional KMeans clustering using optimized KMeans and MiniBatch KMeans,
reducing dimensionality using t-SNE.
"""

import logging
from typing import Optional, Union
import pandas as pd
from fastapi import UploadFile
from sklearn.manifold import TSNE
from app.services.custom_kmeans import OptimizedKMeans, OptimizedMiniBatchKMeans
from app.models.custom_kmeans_model import BasicKMeansResult, KMeansResult3D
from app.services.utils import (process_uploaded_file, 
                                normalize_dataframe,
                                handle_categorical_data, 
                                transform_to_2d_cluster_model, 
                                transform_to_3d_cluster_model)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pylint: disable=too-many-arguments


def perform_nd_kmeans_from_file(
        file: UploadFile,
        distance_metric: str,
        kmeans_type: str,
        user_id: int,
        request_id: int,
        selected_columns: Union[None, list[int]] = None,
        user_k: Optional[int] = None,
        use_3d_model: bool = False
) -> Union[BasicKMeansResult, KMeansResult3D]:
    """
    Perform N-Dimensional KMeans clustering on an uploaded 
    file and reduce dimensionality using t-SNE.
    """
    data_frame, filename = process_uploaded_file(file, selected_columns)
    data_frame = handle_categorical_data(data_frame)
    return _perform_nd_kmeans(data_frame, filename, distance_metric,
                              kmeans_type, user_id, request_id, user_k, use_3d_model)


# pylint: disable=too-many-arguments
def perform_nd_kmeans_from_dataframe(
        data_frame: pd.DataFrame,
        filename: str,
        distance_metric: str,
        kmeans_type: str,
        user_id: int,
        request_id: int,
        user_k: Optional[int] = None,
        use_3d_model: bool = False
) -> Union[BasicKMeansResult, KMeansResult3D]:
    """
    Perform N-Dimensional KMeans clustering on a DataFrame and reduce dimensionality using t-SNE.
    """
    data_frame = handle_categorical_data(data_frame)
    return _perform_nd_kmeans(data_frame, filename, distance_metric,
                              kmeans_type, user_id, request_id, user_k, use_3d_model)


def _perform_nd_kmeans(
        data_frame: pd.DataFrame,
        filename: str,
        distance_metric: str,
        kmeans_type: str,
        user_id: int,
        request_id: int,
        k: int,
        use_3d_model: bool = False
) -> Union[BasicKMeansResult, KMeansResult3D]:

    data_frame = normalize_dataframe(data_frame)
    data_np = data_frame.values

    components = 3 if use_3d_model else 2
    tsne = TSNE(n_components=components)
    data_transformed = tsne.fit_transform(data_np)

    if use_3d_model:
        data_frame = pd.DataFrame(data_transformed, columns=[
                                  't-SNE1', 't-SNE2', 't-SNE3'])
    else:
        data_frame = pd.DataFrame(
            data_transformed, columns=['t-SNE1', 't-SNE2'])

    if kmeans_type == "OptimizedKMeans":
        model = OptimizedKMeans(k, distance_metric)
    elif kmeans_type == "OptimizedMiniBatchKMeans":
        model = OptimizedMiniBatchKMeans(k, distance_metric)
    else:
        raise ValueError(f"Invalid kmeans_type: {kmeans_type}")

    logger.info(data_transformed[:10])

    model.fit(data_transformed)
    data_frame['cluster'] = model.assign_labels(data_transformed)

    if use_3d_model:
        clusters = transform_to_3d_cluster_model(
            data_frame, model.cluster_centers_)
        return KMeansResult3D(
            user_id=user_id,
            request_id=request_id,
            cluster=clusters,
            x_label="t-SNE1",
            y_label="t-SNE2",
            z_label="t-SNE3",
            iterations=model.iterations_,
            used_distance_metric=distance_metric,
            name=filename,
            k_value=k
        )

    clusters = transform_to_2d_cluster_model(
        data_frame, model.cluster_centers_)
    return BasicKMeansResult(
        user_id=user_id,
        request_id=request_id,
        cluster=clusters,
        x_label="t-SNE1",
        y_label="t-SNE2",
        iterations=model.iterations_,
        used_distance_metric=distance_metric,
        name=filename,
        k_value=k
    )
