import pytest
import cv2
import numpy as np
from unittest import mock
import os
import sys
from scripts.camera_calibration import main

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_video_capture():
    with mock.patch('cv2.VideoCapture') as mock_capture:
        mock_instance = mock.Mock()
        mock_capture.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_imshow():
    with mock.patch('cv2.imshow') as mock_show:
        yield mock_show

@pytest.fixture
def mock_waitkey():
    with mock.patch('cv2.waitKey') as mock_key:
        yield mock_key

@pytest.fixture
def mock_find_chessboard_corners():
    with mock.patch('cv2.findChessboardCorners') as mock_corners:
        yield mock_corners

@pytest.fixture
def mock_corner_subpix():
    with mock.patch('cv2.cornerSubPix') as mock_subpix:
        yield mock_subpix

@pytest.fixture
def mock_calibrate_camera():
    with mock.patch('cv2.calibrateCamera') as mock_calibrate:
        yield mock_calibrate

def test_camera_not_opened(mock_video_capture):
    mock_video_capture.isOpened.return_value = False
    with mock.patch('builtins.print') as mock_print:
        main((8, 6), 2.5)
        mock_print.assert_called_with("Error: Could not open video capture")

def test_frame_not_read(mock_video_capture):
    mock_video_capture.isOpened.return_value = True
    mock_video_capture.read.return_value = (False, None)
    with mock.patch('builtins.print') as mock_print:
        main((8, 6), 2.5)
        mock_print.assert_any_call("Error: Could not read frame")

def test_find_chessboard_corners(mock_video_capture, mock_find_chessboard_corners, mock_imshow, mock_waitkey):
    mock_video_capture.isOpened.return_value = True
    mock_video_capture.read.return_value = (True, np.zeros((480, 640, 3), np.uint8))
    mock_find_chessboard_corners.return_value = (True, np.zeros((48, 1, 2), np.float32))
    mock_waitkey.side_effect = [ord('q')]

    with mock.patch('builtins.print') as mock_print:
        main((8, 6), 2.5)
        # Check that print was not called with any error messages
        for call in mock_print.mock_calls:
            if isinstance(call[1][0], str):
                assert "Error" not in call[1][0]

def test_calibrate_camera(mock_video_capture, mock_find_chessboard_corners, mock_corner_subpix, mock_calibrate_camera, mock_imshow, mock_waitkey):
    mock_video_capture.isOpened.return_value = True
    mock_video_capture.read.return_value = (True, np.zeros((480, 640, 3), np.uint8))
    mock_find_chessboard_corners.return_value = (True, np.zeros((48, 1, 2), np.float32))
    mock_corner_subpix.return_value = np.zeros((48, 1, 2), np.float32)
    mock_waitkey.side_effect = [ord('c'), ord('q')]

    mock_calibrate_camera.return_value = (True, np.eye(3), np.zeros(5), [np.zeros((3, 1))], [np.zeros((3, 1))])

    with mock.patch('builtins.print') as mock_print:
        main((8, 6), 2.5)
        # Check that the correct messages are printed
        printed_strings = [call[1][0] for call in mock_print.mock_calls if isinstance(call[1][0], str)]
        assert "Camera matrix:" in printed_strings
        assert "\nDistortion coefficients:" in printed_strings
        assert "\nRotation Vectors:" in printed_strings
        assert "\nTranslation Vectors:" in printed_strings
        # Check that the printed arrays match expected values
        printed_arrays = [call[1][0] for call in mock_print.mock_calls if isinstance(call[1][0], np.ndarray)]
        assert any(np.array_equal(np.eye(3), arr) for arr in printed_arrays)
        assert any(np.array_equal(np.zeros(5), arr) for arr in printed_arrays)