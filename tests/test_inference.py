import json

from fastapi.testclient import TestClient
from online_inference.main import app

client = TestClient(app)
EXAMPLE_PATH = 'example.json'


def test_read_main():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "Ok"}


def test_predict():
    with open(EXAMPLE_PATH, 'r') as f:
        data = f.read()
    response = client.post("/predict", data)
    assert response.status_code == 201
    assert response.json()['predict'] in [0, 1]
