import pytest
import sys
import numpy as np
import os
import cv2
import tempfile

sys.path.append("..")
from tools.process_img import preprocess_img

def test_preprocess_img_with_file_path():
    # Create a temporary image file
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)  # Blue image in BGR
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name

    # Test preprocess_img with file path
    result = preprocess_img(tmp_path, img_size=(512, 512))
    assert isinstance(result, np.ndarray)
    assert result.shape == (512, 512, 3)
    # Check if the color channels are swapped to RGB
    assert np.all(result[..., 0] == 0)  # R
    assert np.all(result[..., 1] == 0)  # G
    assert np.all(result[..., 2] == 255)  # B

    os.remove(tmp_path)

def test_preprocess_img_with_ndarray():
    img = np.ones((300, 400, 3), dtype=np.uint8) * 127
    result = preprocess_img(img, img_size=(256, 256))
    assert isinstance(result, np.ndarray)
    assert result.shape == (256, 256, 3)
    assert np.all(result >= 127 - 1) and np.all(result <= 127 + 1)


