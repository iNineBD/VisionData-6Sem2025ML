import pytest
from src.services.serviceHello import hello

def test_hello_function():
    result = hello()
    assert result == "Hello, VisionData!"