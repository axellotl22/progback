"""
This module provides functions for loading, cleaning and processing dataframes.
"""

import logging
import os
from typing import List
from fastapi import HTTPException
import pandas as pd

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

    raise ValueError(f"Unsupported file type: {os.path.splitext(file_path)[1]}")


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
