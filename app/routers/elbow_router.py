"""
elbow_router.py
---------------

FastAPI router for the KMeans elbow method endpoint.
"""

import os
import logging
from fastapi import APIRouter, UploadFile, HTTPException, Form
from app.models.elbow_model import ElbowResult
from app.services.elbow_service import (run_standard_elbow_method, 
                                        run_optimized_elbow_method)
from app.services.utils import save_temp_file

router = APIRouter()

@router.post("/elbow", response_model=ElbowResult)
async def elbow_method(file: UploadFile, method: str = Form("optimized")):
    """
    Accepts a file and calculates the optimal number of clusters using the elbow method.
    
    Args:
    - file (UploadFile): The file to be processed.
    - method (str): The method to use ("standard" or "optimized").
    
    Returns:
    - ElbowVisualizationResult: JSON representation of the Elbow method.
    """
    
    # Save the uploaded file to a temporary location
    temp_file_path = save_temp_file(file, directory="/tmp")
    
    try:
        # Select the processing method based on user input
        if method == "standard":
            result = run_standard_elbow_method(temp_file_path)
        elif method == "optimized":
            result = run_optimized_elbow_method(temp_file_path)
        else:
            raise HTTPException(
                detail="Invalid method provided. Please choose 'standard' or 'optimized'.", 
                status_code=400)
        
        # Return the result
        return result

    except ValueError as error:
        logging.error("Error reading file: %s", error)
        raise HTTPException(400, "Unsupported file type") from error

    except Exception as error:
        logging.error("Error processing file: %s", error)
        raise HTTPException(500, "Error processing file") from error

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
