import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from model import TicketClassifier


@pytest.fixture
def trained_classifier():
    """Train a small but valid classifier for testing."""
    clf = TicketClassifier()
    texts = [
        # Hardware (5 samples)
        "My laptop screen is flickering badly",
        "The keyboard keys are not responding",
        "My mouse cursor freezes randomly",
        "Laptop battery drains very fast",
        "USB ports not detecting any devices",
        # Network (5 samples)
        "Cannot connect to the internet",
        "WiFi keeps dropping every few minutes",
        "VPN is not connecting at all",
        "Network drive is not accessible",
        "Internet speed is very slow today",
        # Software (5 samples)
        "Microsoft Office crashes on startup",
        "Application throws error when opening",
        "Browser keeps crashing with many tabs",
        "Software update failed and broke the app",
        "Email client is not syncing messages",
        # Account (5 samples)
        "I forgot my password and cannot login",
        "My account has been locked out",
        "Need to reset two-factor authentication",
        "Login credentials not being accepted",
        "My account permissions need updating",
        # Security (5 samples)
        "I received a suspicious phishing email",
        "My computer may have malware on it",
        "There is unauthorized access in my logs",
        "Antivirus detected and quarantined a file",
        "I think my password has been stolen",
    ]
    labels = [
        "Hardware", "Hardware", "Hardware", "Hardware", "Hardware",
        "Network",  "Network",  "Network",  "Network",  "Network",
        "Software", "Software", "Software", "Software", "Software",
        "Account",  "Account",  "Account",  "Account",  "Account",
        "Security", "Security", "Security", "Security", "Security",
    ]
    clf.train(texts, labels)
    return clf


def test_model_trains(trained_classifier):
    assert trained_classifier.is_trained is True


def test_predict_returns_label(trained_classifier):
    result = trained_classifier.predict("My keyboard stopped working")
    assert "label" in result
    assert "confidence" in result
    assert result["label"] in ["Hardware", "Network", "Software", "Account", "Security"]


def test_confidence_range(trained_classifier):
    result = trained_classifier.predict("Cannot log into my account")
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_before_train_raises():
    clf = TicketClassifier()
    with pytest.raises(RuntimeError):
        clf.predict("some text")


def test_save_and_load(trained_classifier, tmp_path):
    save_path = str(tmp_path / "test_model")
    trained_classifier.save(save_path)

    new_clf = TicketClassifier()
    new_clf.load(save_path)

    result = new_clf.predict("My screen is flickering")
    assert "label" in result
    assert result["label"] in ["Hardware", "Network", "Software", "Account", "Security"]