import os
from fastapi import APIRouter, UploadFile, HTTPException, Form
from app.models.elbow_model import ElbowVisualizationResult
from app.services.elbow_service import process_file_for_elbow_method, process_file_for_elbow_method_optimized
from app.services.utils import save_temp_file

router = APIRouter()

@router.post("/elbow", response_model=ElbowVisualizationResult)
async def elbow_method(file: UploadFile, method: str = Form("optimized")):
    """
    Accepts a file and calculates the optimal number of clusters using the elbow method.
    
    Args:
    - file (UploadFile): The file to be processed.
    - method (str): The method to use ("standard" or "optimized").
    
    Returns:
    - ElbowVisualizationResult: JSON representation of the Elbow method results suitable for visualization.
    """
    
    # Save the uploaded file to a temporary location
    temp_file_path = save_temp_file(file, directory="/tmp")
    
    try:
        # Select the processing method based on user input
        if method == "standard":
            result = process_file_for_elbow_method(temp_file_path)
        elif method == "optimized":
            result = process_file_for_elbow_method_optimized(temp_file_path)
        else:
            raise HTTPException(detail="Invalid method provided. Please choose 'standard' or 'optimized'.", status_code=400)
        
        # Return the result
        return result

    except Exception as e:
        raise HTTPException(detail=str(e), status_code=400)

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
