"""
elbow_models.py
---------------

Pydantic models for the elbow method in clustering visualization.
"""

from typing import List
from pydantic import BaseModel

class AxisLabel(BaseModel):
    """
    Labels for plot axes.

    Attributes:
        x (str): Label for x-axis.
        y (str): Label for y-axis.
    """
    x: str
    y: str


class DataPoint(BaseModel):
    """
    A single visualization data point.

    Attributes:
        x (float): Value on x-axis.
        y (float): Value on y-axis.
    """
    x: float
    y: float


class ElbowResult(BaseModel):
    """
    Results of the elbow method for clustering.

    Attributes:
        points (List[DataPoint]): (x, y) points sequence.
        labels (AxisLabel): Plot axis descriptions.
        recommended_point (DataPoint): Optimal k and distortion.
    """
    
    points: List[DataPoint]
    labels: AxisLabel
    recommended_point: DataPoint
