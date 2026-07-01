"""
tests/test_model.py — Unit tests for KidneyScan AI
====================================================
Run with:  pytest tests/ -v
"""

import sys
import os
import pytest
import torch
import numpy as np

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model import (
    build_model,
    KidneyDiseaseNet,
    count_parameters,
    NUM_CLASSES,
    CLASS_NAMES,
    IMAGE_SIZE,
)
from config import IMAGENET_MEAN, IMAGENET_STD


# ─── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def model():
    """Build a fresh model for all tests."""
    return build_model()


@pytest.fixture(scope="module")
def dummy_batch():
    """Return a small random batch of shape (2, 3, 224, 224)."""
    return torch.randn(2, 3, *IMAGE_SIZE)


# ─── Model Structure Tests ─────────────────────────────────────────────────────
class TestModelArchitecture:

    def test_model_is_kidney_disease_net(self, model):
        assert isinstance(model, KidneyDiseaseNet)

    def test_class_names_count(self):
        assert len(CLASS_NAMES) == NUM_CLASSES == 4

    def test_expected_class_names(self):
        assert set(CLASS_NAMES) == {"Cyst", "Normal", "Stone", "Tumor"}

    def test_parameter_count(self, model):
        params = count_parameters(model)
        assert params["total"] > 20_000_000, "ResNet50 should have >20M params"
        assert params["trainable"] > 0, "Some params must be trainable"
        assert params["frozen"] > params["trainable"], "Most params should be frozen"

    def test_image_size(self):
        assert IMAGE_SIZE == (224, 224)


# ─── Forward Pass Tests ────────────────────────────────────────────────────────
class TestForwardPass:

    def test_output_shape(self, model, dummy_batch):
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (2, NUM_CLASSES), f"Expected (2,{NUM_CLASSES}), got {out.shape}"

    def test_output_not_nan(self, model, dummy_batch):
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert not torch.isnan(out).any(), "Model output contains NaN"

    def test_softmax_sums_to_one(self, model, dummy_batch):
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(dummy_batch), dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5), \
            f"Softmax does not sum to 1: {sums}"

    def test_probabilities_range(self, model, dummy_batch):
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(dummy_batch), dim=1)
        assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of [0,1] range"

    def test_predict_proba_method(self, model, dummy_batch):
        probs = model.predict_proba(dummy_batch)
        assert probs.shape == (2, NUM_CLASSES)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


# ─── Preprocessing Tests ──────────────────────────────────────────────────────
class TestPreprocessing:

    def test_imagenet_mean_length(self):
        assert len(IMAGENET_MEAN) == 3

    def test_imagenet_std_length(self):
        assert len(IMAGENET_STD) == 3

    def test_imagenet_std_nonzero(self):
        for s in IMAGENET_STD:
            assert s > 0, "Std should be positive"


# ─── Config Tests ─────────────────────────────────────────────────────────────
class TestConfig:

    def test_config_imports(self):
        from config import (
            ROOT_DIR, MODEL_PATH, CLASS_NAMES, NUM_CLASSES,
            BATCH_SIZE, EPOCHS, LEARNING_RATE,
        )
        assert NUM_CLASSES == 4
        assert BATCH_SIZE > 0
        assert EPOCHS > 0
        assert 0 < LEARNING_RATE < 1
