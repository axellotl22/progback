""" Router für Classification-Endpunkte. """

import os
import logging

from typing import Optional, List, Union
from sklearn.model_selection import train_test_split
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.classification_decision_tree_model import TreeNode, DecisionTreeData, DecisionTreeResult, SplitStrategy, DecisionTree
import app.services.classification_decision_tree_service
import app.models.classification_decision_tree_model
from app.services.utils import clean_dataframe, select_columns, delete_file, load_dataframe


TEST_MODE = os.environ.get("TEST_MODE", "False") == "True"
TEMP_FILES_DIR = "temp_files/"

router = APIRouter()
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
@router.post("/perform-decision_tree-classification/", response_model=DecisionTreeResult)
async def perform_decision_tree_classification(
    file: UploadFile = File(...),
    columns: Optional[Union[str, List[int]]] = None
):
    """
    Dieser Endpunkt verarbeitet die hochgeladene Datei und gibt 
    den Entscheidungsbaum zurück. Der Benutzer kann optional 
    die zu berücksichtigenden Spalten bestimmen.
    """
    
    if not os.path.exists(TEMP_FILES_DIR):
        os.makedirs(TEMP_FILES_DIR)

    file_path = os.path.join(TEMP_FILES_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    try:
        data_frame = load_dataframe(file_path)
        data_frame = clean_dataframe(data_frame)

        if columns:
            data_frame = select_columns(data_frame, columns)

        df = convert_text_to_categorical(df)
        target_column=("Drug")
        #target_column="Education"

        # Features und Labels extrahieren
        X = df.drop(target_column, axis=1).values  # Target Column Name = Zu klassifizierende Spalte
        y = df[target_column].values

        # Feature-Namen extrahieren
        #feature_names = df.columns[:-1].tolist()
        #data = datasets.load_breast_cancer()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        clf = DecisionTree(split_strategy=SplitStrategy.DURCHSCHNITT)
        clf.fit(X_train, y_train)
        return DecisionTreeResult(clf)
    except ValueError as error:
        logging.error("Fehler beim Lesen der Datei: %s", error)
        raise HTTPException(400, "Nicht unterstützter Dateityp") from error

    except Exception as error:
        logging.error("Fehler bei der Dateiverarbeitung: %s", error)
        raise HTTPException(500, "Fehler bei der Dateiverarbeitung") from error

    finally:
        if not TEST_MODE:
            delete_file(file_path)
