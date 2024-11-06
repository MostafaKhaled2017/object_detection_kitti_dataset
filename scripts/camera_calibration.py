import cv2
import numpy as np
import os
import argparse

def main(checkerboard_size, square_size):
    # Define parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    camera_index = 0
    output_directory = 'data/checkerboard'

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    # Start capturing images from the camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            frame = cv2.drawChessboardCorners(frame, checkerboard_size, corners2, ret)

        cv2.imshow('Calibration', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c') and ret:
            filename = os.path.join(output_directory, f'calibration_image_{len(imgpoints)}.png')
            cv2.imwrite(filename, frame)
            print(f'Saved calibration image {len(imgpoints)} at {filename}')

    cap.release()
    cv2.destroyAllWindows()

    # Perform camera calibration
    if len(objpoints) > 0 and len(imgpoints) > 0:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv2.CALIB_USE_LU)
        if ret:
            print("Camera matrix:")
            print(camera_matrix)
            print("\nDistortion coefficients:")
            print(dist_coeffs)
            print("\nRotation Vectors:")
            print(rvecs)
            print("\nTranslation Vectors:")
            print(tvecs)
        else:
            print("Calibration failed")
    else:
        print("Not enough data for calibration")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration using checkerboard images.")
    parser.add_argument('--checkerboard_size', type=int, nargs=2, default=(8, 6), help='Number of inner corners per a checkerboard row and column.')
    parser.add_argument('--square_size', type=float, default=2.5, help='Square size in centimeter square.')
    args = parser.parse_args()

    main(tuple(args.checkerboard_size), args.square_size)