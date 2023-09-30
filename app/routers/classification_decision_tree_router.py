""" Router für Clustering-Endpunkte. """

import os
import logging

from typing import Optional, List, Union

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.classification_decision_tree_model import TreeNode, DecisionTreeData, DecisionTreeResult
import app.services.classification_decision_tree_service
from app.services.utils import clean_dataframe, select_columns
import app.services.utils

TEST_MODE = os.environ.get("TEST_MODE", "False") == "True"
TEMP_FILES_DIR = "temp_files/"

router = APIRouter()

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
    if columns:
        columns = process_columns_input(columns)

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

        decision_tree = create_tree(data_frame)

        return DecisionTreeResult(tree=decision_tree)
    except ValueError as error:
        logging.error("Fehler beim Lesen der Datei: %s", error)
        raise HTTPException(400, "Nicht unterstützter Dateityp") from error

    except Exception as error:
        logging.error("Fehler bei der Dateiverarbeitung: %s", error)
        raise HTTPException(500, "Fehler bei der Dateiverarbeitung") from error

    finally:
        if not TEST_MODE:
            delete_file(file_path)
