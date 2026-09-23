from app import classify_image


def test_classify_image():
    assert classify_image() == "Image classification system is working"
