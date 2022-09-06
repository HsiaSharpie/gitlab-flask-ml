import json
import pickle
from sklearn.datasets import make_regression

from app import app


def test_api():
    client = app.test_client()
    url = "http://localhost:5000/api/v1/model"
    mock_request_data = {
        "payload": [
            -0.8304850047899216,
            0.6574702801539302,
            -0.48461637843134797,
            -0.6719012158053788,
            -0.4059397149127815,
            -0.2658320744432613,
            -0.010997830598481148,
            1.3059189718132382,
            -0.5943064070116102,
            1.7601385092595263,
            0.3452747986164928
        ]
    }    
    response = client.post(url, data=json.dumps(mock_request_data))
    assert response.status_code == 200