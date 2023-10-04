"""
elbow_models.py
---------------

Pydantic models for the elbow method visualization. 
"""

from typing import List
from pydantic import BaseModel

class AxisLabel(BaseModel):
    """
    Description of the plot axes.

    Attributes:
        x (str): Label for x-axis 
        y (str): Label for y-axis
    """
    x: str
    y: str


class DataPoint(BaseModel):
    """
    Single data point for visualization.

    Attributes:
        x (float): X-axis value
        y (float): Y-axis value   
    """
    x: float
    y: float


class ElbowResult(BaseModel):
    """
    Response model for elbow method endpoint.

    Attributes:
        points (List[DataPoint]): List of (x, y) points 
        labels (AxisLabel): Axis labels
        recommended_point (DataPoint): Recommended k value and distortion
    """
    
    points: List[DataPoint]
    labels: AxisLabel
    recommended_point: DataPoint
