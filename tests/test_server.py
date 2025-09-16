import json
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_users_and_prefs_cycle(tmp_path):
    # create user
    r = client.post("/users", json={"username": "alice"})
    assert r.status_code == 200
    user_id = r.json()["user_id"]

    # save prefs
    params = {"dataset": "dataset_01", "days": 5, "n_lags": 10, "epochs": 1, "learning_rate": 0.001, "batch_size": 8, "number_nodes": 16}
    r = client.post("/preferences", json={"user_id": user_id, "params": params})
    assert r.status_code == 200

    # load prefs
    r = client.get(f"/preferences/{user_id}")
    assert r.status_code == 200
    assert r.json()["params"]["dataset"] == "dataset_01"
