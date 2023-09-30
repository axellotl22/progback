"""
Dienste für Decision Tree Classification-Funktionen.
"""

import ast
import logging
import os
from typing import List, Union

import numpy as np
import pandas as pd
from fastapi import HTTPException
from gap_statistic import OptimalK
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# Logging-Einstellungen
logging.basicConfig(level=logging.INFO)

MAX_CLUSTERS = 10


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Lädt eine Datei in ein Pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    if file_path.endswith('.json'):
        return pd.read_json(file_path)
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)

    raise ValueError(
        f"Unsupported file type: {os.path.splitext(file_path)[1]}")

def clean_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt das DataFrame von leeren und unvollständigen Zeilen.
    """
    return data_frame.dropna()

