"""
tests/test_api.py — Flask API unit tests
========================================
Run with:  pytest tests/ -v
"""

import sys
import os
import io
import json
import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Create test client ───────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """Return a Flask test client."""
    from api.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ─── Helper: create in-memory PNG image ───────────────────────────────────────
def _make_png_bytes(w: int = 224, h: int = 224) -> bytes:
    arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ─── Health endpoint ──────────────────────────────────────────────────────────
class TestHealthEndpoint:

    def test_health_status_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_returns_json(self, client):
        resp = client.get("/api/health")
        data = json.loads(resp.data)
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_has_classes(self, client):
        resp = client.get("/api/health")
        data = json.loads(resp.data)
        assert "classes" in data
        assert len(data["classes"]) == 4


# ─── Model info endpoint ──────────────────────────────────────────────────────
class TestModelInfoEndpoint:

    def test_model_info_200(self, client):
        resp = client.get("/api/model-info")
        assert resp.status_code == 200

    def test_model_info_fields(self, client):
        resp = client.get("/api/model-info")
        data = json.loads(resp.data)
        assert "model_name" in data
        assert "classes" in data
        assert "input_size" in data


# ─── Predict endpoint ─────────────────────────────────────────────────────────
class TestPredictEndpoint:

    def test_predict_no_file_returns_400(self, client):
        resp = client.post("/api/predict")
        assert resp.status_code == 400

    def test_predict_valid_image_returns_200(self, client):
        png_bytes = _make_png_bytes()
        data = {"image": (io.BytesIO(png_bytes), "test_ct_scan.png")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200

    def test_predict_response_has_class(self, client):
        png_bytes = _make_png_bytes()
        data = {"image": (io.BytesIO(png_bytes), "scan.png")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        result = json.loads(resp.data)
        assert "predicted_class" in result
        assert result["predicted_class"] in ["Cyst", "Normal", "Stone", "Tumor"]

    def test_predict_response_has_probabilities(self, client):
        png_bytes = _make_png_bytes()
        data = {"image": (io.BytesIO(png_bytes), "scan.png")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        result = json.loads(resp.data)
        assert "probabilities" in result
        probs = result["probabilities"]
        assert len(probs) == 4
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities should sum to ~1, got {total}"

    def test_predict_confidence_in_range(self, client):
        png_bytes = _make_png_bytes()
        data = {"image": (io.BytesIO(png_bytes), "scan.png")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        result = json.loads(resp.data)
        conf = result["confidence"]
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_predict_has_description(self, client):
        png_bytes = _make_png_bytes()
        data = {"image": (io.BytesIO(png_bytes), "scan.png")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        result = json.loads(resp.data)
        assert "description" in result and len(result["description"]) > 10

    def test_predict_invalid_extension_returns_400(self, client):
        data = {"image": (io.BytesIO(b"fake"), "scan.txt")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400


# ─── Static UI ────────────────────────────────────────────────────────────────
class TestStaticUI:

    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_is_html(self, client):
        resp = client.get("/")
        assert b"KidneyScan" in resp.data or b"<!DOCTYPE" in resp.data
