from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    """
        Item Klasse
    """
    id: int
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: list[str] = []


@app.get("/")
def hello_world():
    """
    Hello World

    :return:
    """
    return {"hello": "test deploy"}