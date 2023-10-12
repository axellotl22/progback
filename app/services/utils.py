"""
This module provides functions for loading, cleaning and processing dataframes.
"""

import logging
import os
from typing import List, Union
import numpy as np

from fastapi import HTTPException, UploadFile
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.models.basic_kmeans_model import Cluster, Centroid, Cluster3D, Centroid3D

# Constants in uppercase
CSV = '.csv'
XLSX = '.xlsx'
XLS = '.xls'
JSON = '.json'

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Loads file into a Pandas DataFrame.

    Args:
        file_path (str): File path

    Returns:
        pd.DataFrame: DataFrame of loaded data

    Raises:
        ValueError: If file type is not supported
    """
    # Load dataframe based on file type
    if file_path.endswith(CSV):
        return pd.read_csv(file_path)

    if file_path.endswith(XLSX) or file_path.endswith(XLS):
        return pd.read_excel(file_path)

    if file_path.endswith(JSON):
        return pd.read_json(file_path)

    raise ValueError(
        f"Unsupported file type: {os.path.splitext(file_path)[1]}")


def clean_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Removes empty and incomplete rows.

    Args:
        data_frame (pd.DataFrame): DataFrame to clean

    Returns:
        pd.DataFrame: Cleaned dataframe
    """

    # Convert columns with only 2 unique values to bool
    for col in data_frame.columns:
        if len(data_frame[col].unique()) == 2:
            data_frame[col] = data_frame[col].astype(bool)

    # Drop empty rows
    return data_frame.dropna()


def select_columns(data_frame: pd.DataFrame, columns: List[int]) -> pd.DataFrame:
    """Selects columns by their indices.

    Args:
        data_frame (pd.DataFrame): DataFrame
        columns (List[int]): List of column indices  

    Returns:
        pd.DataFrame: DataFrame with selected columns

    Raises:
        ValueError: If invalid column index
    """
    # Validate column indices
    if any(col_idx >= len(data_frame.columns) for col_idx in columns):
        raise ValueError(
            f"Invalid column index. DataFrame has only {len(data_frame.columns)} columns.")

    # Select columns by index
    selected_cols = [data_frame.columns[idx] for idx in columns]
    return data_frame[selected_cols]


def extract_selected_columns(data_frame: pd.DataFrame,
                             selected_columns: Union[None, list[int]] = None) -> pd.DataFrame:
    """
    Extract the specified columns based on their indices from the dataframe.
    """
    if selected_columns:
        columns_to_select = [data_frame.columns[i] for i in selected_columns]
        return data_frame[columns_to_select]
    return data_frame


def delete_file(file_path: str):
    """Deletes specified file.

    Args:
        file_path (str): File path
    """
    try:
        if os.environ.get("TEST_MODE") != "True":

            # Actually delete file
            os.remove(file_path)

            logger.info("File %s successfully deleted.", file_path)

    except FileNotFoundError:

        # File already deleted
        logger.warning("File %s already deleted.", file_path)

    except OSError as err:

        # Handle other errors
        logger.error("Error deleting %s: %s", file_path, err)


def save_temp_file(file, directory):
    """Saves a temporary file and returns the path."""

    # Create directory if needed
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate file path
    file_path = os.path.join(directory, file.filename)

    # Write file contents
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return file_path


def handle_errors(error):
    """
    Error handling function.
    """
    if isinstance(error, ValueError):
        logging.error("Error reading file: %s", error)
        raise HTTPException(400, "Unsupported file type") from error
    logging.error("Error processing file: %s", error)
    raise HTTPException(500, "Error processing file") from error


def process_uploaded_file(file: UploadFile,
                          selected_columns: Union[None, list[int]] = None) -> (pd.DataFrame, str):
    """
    Load, save, clean, and optionally select specific columns from the uploaded file. 
    Returns the cleaned dataframe and filename.

    Args:
    - file (UploadFile): Uploaded data file.
    - selected_columns (list[int], optional): Indices of selected columns, if any.

    Returns:
    - tuple: Cleaned dataframe and the filename of the uploaded file.
    """
    temp_file_path = save_temp_file(file, "temp/")
    data_frame = load_dataframe(temp_file_path)
    data_frame = clean_dataframe(data_frame)

    # Select specific columns if provided
    if selected_columns:
        data_frame = extract_selected_columns(data_frame, selected_columns)

    delete_file(temp_file_path)
    return data_frame, file.filename


def handle_categorical_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical and boolean columns to numerical format using one-hot encoding.
    """
    return pd.get_dummies(data_frame, drop_first=True)


def normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the data using StandardScaler.
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(dataframe)
    normalized_df = pd.DataFrame(normalized_data, columns=dataframe.columns)
    return normalized_df


def transform_to_2d_cluster_model(data_frame: pd.DataFrame, cluster_centers: np.ndarray) -> list:
    """
    Transform the data into the Cluster model structure.
    """
    clusters_list = []

    for cluster_id in range(cluster_centers.shape[0]):
        cluster_data = data_frame[data_frame["cluster"] == cluster_id].drop(columns=[
                                                                            "cluster"])

        # Transform points to always have "x" and "y" as keys
        cluster_points = [{"x": row.iloc[0], "y": row.iloc[1]}
                          for _, row in cluster_data.iterrows()]

        clusters_list.append(
            Cluster(
                clusterNr=cluster_id,
                centroid=Centroid(
                    x=cluster_centers[cluster_id][0], y=cluster_centers[cluster_id][1]),
                points=cluster_points
            )
        )

    return clusters_list


def transform_to_3d_cluster_model(data_frame: pd.DataFrame, cluster_centers: np.ndarray) -> list:
    """
    Transform the data into the 3D Cluster model structure.
    """
    clusters_list = []

    for cluster_id in range(cluster_centers.shape[0]):
        cluster_data = data_frame[data_frame["cluster"] == cluster_id].drop(columns=[
                                                                            "cluster"])

        # Transform points to always have "x", "y", and "z" as keys
        cluster_points = [{"x": row.iloc[0], "y": row.iloc[1], "z": row.iloc[2]}
                          for _, row in cluster_data.iterrows()]

        clusters_list.append(
            Cluster3D(
                clusterNr=cluster_id,
                centroid=Centroid3D(
                    x=cluster_centers[cluster_id][0],
                    y=cluster_centers[cluster_id][1],
                    z=cluster_centers[cluster_id][2]),
                points=cluster_points
            )
        )

    return clusters_list
