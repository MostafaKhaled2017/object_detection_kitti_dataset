import open3d as o3d
import numpy as np
import os
import sys
import random
import argparse

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def read_kitti_bin_point_cloud(bin_path):
    """ Read KITTI bin file and convert to Open3D point cloud """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

def downsample_point_cloud(pcd, voxel_size=0.05):
    """ Downsample the point cloud using a voxel grid filter """
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return down_pcd

def estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)):
    """ Estimate normals in the point cloud """
    pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_consistent_tangent_plane(100)
    return pcd

def augment_data(pcd):
    """ Data augmentation through random rotations """
    R = pcd.get_rotation_matrix_from_xyz((random.uniform(-np.pi, np.pi), 
                                          random.uniform(-np.pi, np.pi), 
                                          random.uniform(-np.pi, np.pi)))
    pcd.rotate(R, center=(0, 0, 0))
    return pcd

def remove_ground_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """ Remove ground plane from the point cloud using RANSAC """
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    pcd_without_ground = pcd.select_by_index(inliers, invert=True)
    return pcd_without_ground

def save_point_cloud_to_bin(pcd, file_path):
    """ Save Open3D point cloud as a KITTI bin file """
    np_points = np.asarray(pcd.points)
    reflectance = np.zeros((np_points.shape[0], 1), dtype=np.float32)
    np_points = np.hstack((np_points, reflectance))
    np_points.astype(np.float32).tofile(file_path)

def process_point_clouds_in_directory(directory_path):
    """ Process all point clouds in a directory """
    for filename in os.listdir(directory_path):
        if filename.endswith('.bin'):
            bin_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")
            pcd = read_kitti_bin_point_cloud(bin_path)

            pcd = downsample_point_cloud(pcd, voxel_size=0.1)
            pcd = estimate_normals(pcd)
            pcd = augment_data(pcd)
            pcd = remove_ground_plane(pcd)

            # Define processed file path
            processed_file_path = os.path.join(directory_path, 'processed', filename)
            
            # Save processed point cloud
            os.makedirs(os.path.join(directory_path, 'processed'), exist_ok=True)
            save_point_cloud_to_bin(pcd, processed_file_path)
            print(f"Saved processed point cloud to {processed_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process point clouds in a directory.")
    parser.add_argument('--directory_path', type=str, default='data/Kitti_00/velodyne/0000', help='Path to the directory containing point cloud files.')
    args = parser.parse_args()

    process_point_clouds_in_directory(args.directory_path)