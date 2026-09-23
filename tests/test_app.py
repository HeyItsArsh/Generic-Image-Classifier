import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import classify_image


def test_classify_image():
    assert classify_image() == "Image classification system is working"
