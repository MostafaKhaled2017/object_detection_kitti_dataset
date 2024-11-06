import pytest
import numpy as np
import open3d as o3d
import os
from pathlib import Path
import sys

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.point_cloud_preprocessing import (
    read_kitti_bin_point_cloud,
    downsample_point_cloud,
    estimate_normals,
    augment_data,
    remove_ground_plane,
    save_point_cloud_to_bin
)

def test_read_kitti_bin_point_cloud():
    bin_path = 'data/Kitti_00/velodyne/0000/000000.bin'
    points = np.random.rand(100, 4).astype(np.float32)
    points.tofile(bin_path)
    
    pcd = read_kitti_bin_point_cloud(bin_path)
    assert isinstance(pcd, o3d.geometry.PointCloud)
    assert len(pcd.points) == 100


def test_downsample_point_cloud():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))

    down_pcd = downsample_point_cloud(pcd, voxel_size=0.1)
    assert isinstance(down_pcd, o3d.geometry.PointCloud)
    assert len(down_pcd.points) < 1000  # Should be less after downsampling

def test_estimate_normals():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))

    pcd_with_normals = estimate_normals(pcd)
    assert isinstance(pcd_with_normals, o3d.geometry.PointCloud)
    assert np.asarray(pcd_with_normals.normals).shape == (1000, 3)

def test_augment_data():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))

    augmented_pcd = augment_data(pcd)
    assert isinstance(augmented_pcd, o3d.geometry.PointCloud)
    assert np.asarray(augmented_pcd.points).shape == (1000, 3)

def test_remove_ground_plane():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))

    pcd_without_ground = remove_ground_plane(pcd)
    assert isinstance(pcd_without_ground, o3d.geometry.PointCloud)
    assert len(pcd_without_ground.points) < 1000  # Should be less after removing ground plane

def test_save_point_cloud_to_bin():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
    
    bin_path = 'test_saved_point_cloud.bin'
    save_point_cloud_to_bin(pcd, bin_path)
    
    assert os.path.exists(bin_path)
    
    # Read the saved file to check contents
    loaded_points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    assert loaded_points.shape == (1000, 4)
    
    # Clean up the temporary file
    os.remove(bin_path)

if __name__ == "__main__":
    pytest.main()