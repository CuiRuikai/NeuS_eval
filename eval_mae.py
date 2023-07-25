import numpy as np
import open3d as o3d
import point_cloud_utils as pcu


predicted = o3d.io.read_point_cloud('./data/ours.ply')
gt = o3d.io.read_point_cloud('./data/gt_mesh.ply')
points = np.concatenate([np.asarray(predicted.points), np.asarray(gt.points)])

# Assuming `points` is your point cloud, an Nx3 numpy array
center = points.mean(axis=0)
radius = np.max(np.sqrt(np.sum((points - center)**2, axis=1)))

# Number of points to sample
num_samples = 1000

# Sample uniformly within the bounding sphere
u = np.random.uniform(0, 1, num_samples)
v = np.random.uniform(0, 1, num_samples)

theta = 2.0 * np.pi * u  # azimuthal angle
phi = np.arccos(2.0 * v - 1.0)  # polar angle

# Convert spherical coordinates to cartesian
x = radius * np.sin(phi) * np.cos(theta) + center[0]
y = radius * np.sin(phi) * np.sin(theta) + center[1]
z = radius * np.cos(phi) + center[2]
samples = np.stack([x, y, z], axis=-1)


# query signed distance of samples to predicted mesh
v, f = pcu.load_mesh_vf('./data/ours.ply')
sdf_predicted, fid, bc = pcu.signed_distance_to_mesh(samples, v.astype('double'), f)

# query signed distance of samples to gt mesh
v, f = pcu.load_mesh_vf('./data/gt_mesh.ply')
sdf_gt, fid, bc = pcu.signed_distance_to_mesh(samples, v.astype('double'), f)

print(np.abs(sdf_predicted - sdf_gt).mean())