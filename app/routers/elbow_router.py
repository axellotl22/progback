"""
elbow_router.py
---------------

FastAPI router that exposes an endpoint for determining 
the optimal number of clusters using the elbow method.
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
    Accepts an uploaded data file and, based on the selected method, 
    computes and returns the optimal number of clusters using the KMeans elbow method.
    
    Two methods are provided:
    - "standard": Uses the standard K-Means clustering algorithm. This method 
                  is suited for smaller datasets and provides precise cluster assignments.
    - "optimized": Employs the MiniBatchKMeans clustering method combined with 
                   PCA with n_components set to 0.95. This method is optimized 
                   for larger datasets and ensures faster computation. MiniBatchKMeans 
                   speeds up the clustering process by using random samples of the 
                   dataset in each iteration. PCA is employed to reduce dimensionality 
                   and noise, focusing on the main components of the data that capture 
                   the highest variance, aiding in efficiency and potentially improving 
                   cluster quality.
    
    Args:
    - file (UploadFile): The data file uploaded by the user.
    - method (str): The clustering method to use. Two options are available: 
                    "standard" (employs a normal, proprietary KMeans approach) 
                    and "optimized" 
                    (uses MiniBatchKMeans combined with PCA where n_components=0.95).
    
    Returns:
    - ElbowVisualizationResult: A JSON object that includes the optimal number of clusters 
                                and other relevant information from the elbow method analysis.
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
