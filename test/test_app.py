"""
Test implementation
"""
from app.main import app, hello_world, create_item, Item

def test_setup():
    """
    Tests the setup of the API endpoint
    """
    assert app is not None
    assert hello_world is not None
    assert create_item is not None
    assert Item is not None


def test_endpoints():
    """
    Tests the endpoint contract
    """
    res = hello_world()
    assert res['hello'] is not None

    res = create_item(Item(name="Portal Gun", price=42.0, id=1))
    expected = Item(name="Portal Gun", price=42.0, id=1)
    assert res == expected
