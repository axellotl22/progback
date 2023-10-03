from pydantic import BaseModel
from typing import List

class AxisLabel(BaseModel):
    """
    Beschreibung für die Achsen.

    Attributes:
    - x (str): Beschreibung für die x-Achse
    - y (str): Beschreibung für die y-Achse
    """
    x: str
    y: str

class Point(BaseModel):
    """
    Datenpunkt für die Visualisierung.

    Attributes:
    - x (float): Wert der x-Achse
    - y (float): Wert der y-Achse
    """
    x: float
    y: float

class ElbowVisualizationResult(BaseModel):
    """
    Rückgabeobjekt für die Elbow-Methode.

    Attributes:
    - points (List[Point]): Liste der Datenpunkte für x und y Achse
    - labels (AxisLabel): Beschriftungen der Achsen
    - recommended_point (Point): Empfohlener Datenpunkt (k-Wert) mit entsprechendem x und y Wert
    """
    points: List[Point]
    labels: AxisLabel
    recommended_point: Point
