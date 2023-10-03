"""
Services
"""

import logging
import numpy as np
from collections import Counter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def accuracy (y_test, y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

def convert_text_to_categorical(df):
    """
    Convert all text columns in a DataFrame to categorical columns.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - DataFrame with text columns converted to categorical columns
    """
    for col in df.columns:
        if df[col].dtype == 'object':  # if column has text values
            df[col] = df[col].astype('category').cat.codes  # convert to categorical codes
    return df

def sort_data(df, column_index):
    """
    Sortiert die Daten basierend auf einem bestimmten Feature.
    """
    # Sortiert die Indizes von X basierend auf dem gegebenen Feature.
    column_name = df.columns[column_index]
    return df.sort_values(by=column_name)
