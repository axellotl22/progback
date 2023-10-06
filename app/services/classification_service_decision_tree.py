"""
Services
"""
import numpy as np


def accuracy (y_test, y_pred):
    """
    Berechnen der Genauigkeit
    """
    return np.sum(y_test==y_pred)/len(y_test)

def convert_text_to_categorical(data_frame):
    """
    Convert all text columns in a DataFrame to categorical columns.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - DataFrame with text columns converted to categorical columns
    """
    for col in data_frame.columns:
        if data_frame[col].dtype == 'object':  # Falls Textwerte vorhanden
            data_frame[col] = data_frame[col].astype('category').cat.codes
    return data_frame

def sort_data(data_frame, column_index):
    """
    Sortiert die Daten basierend auf einem bestimmten Feature.
    """
    # Sortiert die Indizes von X basierend auf dem gegebenen Feature.
    column_name = data_frame.columns[column_index]
    return data_frame.sort_values(by=column_name)
