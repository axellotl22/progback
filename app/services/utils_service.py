import logging
import os
import pandas as pd
from typing import List

# Logging-Einstellungen
logging.basicConfig(level=logging.INFO)

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

def select_columns(data_frame: pd.DataFrame, columns: List[int]) -> pd.DataFrame:
    """
    Wählt bestimmte Spalten aus einem DataFrame aus basierend auf deren Index.
    """
    if any(col_idx >= len(data_frame.columns) for col_idx in columns):
        raise ValueError(
            f"Ungültiger Spaltenindex. Das DataFrame hat nur {len(data_frame.columns)} Spalten.")

    selected_columns = [data_frame.columns[idx] for idx in columns]
    return data_frame[selected_columns]

def delete_file(file_path: str):
    """
    Löscht die angegebene Datei.
    """
    try:
        if os.environ.get("TEST_MODE") != "True":
            os.remove(file_path)
            logging.info("File %s successfully deleted.", file_path)
    except FileNotFoundError:
        logging.warning("File %s was already deleted.", file_path)
    except OSError as error:
        logging.error("Error deleting file %s: %s", file_path, error)

