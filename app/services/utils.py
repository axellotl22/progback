"""
Dieses Modul bietet Funktionen zum Laden, Bereinigen und Verarbeiten von Datenframes.
"""

import logging
import os
import pandas as pd
from typing import List

# Konstanten in Großbuchstaben
CSV = '.csv'
XLSX = '.xlsx' 
XLS = '.xls'
JSON = '.json'
PARQUET = '.parquet'

# Logging konfigurieren
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Lädt Datei in Pandas DataFrame.

    Args:
        file_path (str): Dateipfad

    Returns:
        pd.DataFrame: DataFrame der geladenen Daten
    
    Raises:
        ValueError: Falls Dateityp nicht unterstützt wird
    """
    if file_path.endswith(CSV):
        return pd.read_csv(file_path)
    
    if file_path.endswith(XLSX) or file_path.endswith(XLS):
        return pd.read_excel(file_path)

    if file_path.endswith(JSON):
        return pd.read_json(file_path)

    if file_path.endswith(PARQUET):
        return pd.read_parquet(file_path)

    raise ValueError(f"Unsupported file type: {os.path.splitext(file_path)[1]}")

def clean_dataframe(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Entfernt leere und unvollständige Zeilen.

    Args:
        data_frame (pd.DataFrame): zu bereinigendes DataFrame

    Returns:
        pd.DataFrame: bereinigtes DataFrame
    """
    return data_frame.dropna()

def select_columns(data_frame: pd.DataFrame, columns: List[int]) -> pd.DataFrame:
    """Wählt Spalten anhand ihrer Indizes aus.

    Args:
        data_frame (pd.DataFrame): DataFrame
        columns (List[int]): Liste der auszuwählenden Spaltenindizes

    Returns:
        pd.DataFrame: DataFrame mit ausgewählten Spalten
    
    Raises:
        ValueError: Falls ungültiger Spaltenindex
    """
    if any(col_idx >= len(data_frame.columns) for col_idx in columns):
        raise ValueError(f"Invalid column index. DataFrame has only {len(data_frame.columns)} columns.")

    selected_cols = [data_frame.columns[idx] for idx in columns]
    return data_frame[selected_cols]

def delete_file(file_path: str):
    """Löscht angegebene Datei.

    Args:
        file_path (str): Dateipfad
    """
    try:
        if os.environ.get("TEST_MODE") != "True":
            os.remove(file_path)
            logger.info("File %s successfully deleted.", file_path)
    except FileNotFoundError:
         logger.warning("File %s already deleted.", file_path)
    except OSError as err:
         logger.error("Error deleting %s: %s", file_path, err)