"""
FastAPI Test
"""
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


@app.get("/hello")
def hello_world():
    """
    Hello World

    :return:
    """
    return {"hello": "test deploy"}


@app.post("/items/")
def create_item(item: Item) -> Item:
    """
    Create Item Test

    :param item:
    :return:
    """
    return item


@app.get("/items/")
def read_items() -> list[Item]:
    """
    List Item Test

    :return:
    """
    return [
        Item(name="Portal Gun", price=42.0, id=1),
        Item(name="Plumbus", price=32.0, id=2),
    ]


@app.get("/items/{id}")
def read_item(item_id: int) -> Item:
    """
    List One Item Test

    :param item_id:
    :return:
    """
    return Item(name="Portal Gun", price=42.0, id=item_id)
