import sys
import numpy as np

from scipy import io
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

import cv2 
from cv2 import DMatch

def parse_config_file(file_path):
    config_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Ignore comments
            if line.startswith('#') or line == '':
                continue

            # Split the line into tokens
            tokens = line.split()

            # Extract parameter names and values
            param_name = tokens[0]
            param_values = [tokens[1:]] if len(tokens) > 2 else tokens[1]
            
            # Check if the token already exists in the dictionary
            if param_name in config_dict:
                # Add new values to the existing token
                config_dict[param_name].extend(param_values)
            else:
                # Create a new entry in the dictionary
                config_dict[param_name] = param_values

    return config_dict

def match_features(features1, features2, matches_size = 100, num_features= 64):
    C = np.vstack((features1, features2))
    
    # PCA
    pca = PCA(n_components=num_features)
    reconstructed = pca.fit_transform(C)
    
    features1 = reconstructed[:len(features1), :]
    features2 = reconstructed[len(features1):, :]
    
    # Euclidean distance
    D = cdist(features1, features2, 'euclidean')
    
    # Sorting distances and finding nearest neighbors
    I = np.argsort(D, axis=1)
    nearest_neighbor = D[np.arange(len(D)), I[:, 0]]
    second_nearest_neighbor = D[np.arange(len(D)), I[:, 1]]
    confidences = nearest_neighbor / second_nearest_neighbor
    
    # Filtering non-zero confidences
    i = np.where(confidences)[0]
    matches = np.column_stack((i, I[i]))
    confidences = 1.0 / confidences[i]
    
    # Sorting by confidence and selecting top 100 matches
    sorted_indices = np.argsort(confidences)[::-1]
    matches = matches[sorted_indices][:matches_size, :]
    confidences = confidences[sorted_indices][:matches_size]

    matches = [DMatch(_queryIdx=int(match[0]), 
                      _trainIdx=int(match[1]), 
                      _distance=float(D[int(match[0]), int(match[1])])) 
                          for match in matches]
    
    return matches, confidences

def create_homography_matrix(src_points, dst_points):
    A = []
    b = [] 
    for i in range(len(src_points)):
        x, y = src_points[i]
        u, v = dst_points[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])

        b.append(u)
        b.append(v)

    A = np.array(A)

    h = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)), b)
    h = np.append(h, 1) 
    h = h.reshape(3,3)
    
    return h

def ransac_homography_custom(src_points, dst_points, iterations=1000, threshold=5.0):
    """
    RANSAC algorithm to compute the homography matrix with outlier rejection, using custom homography computation.

    :param src_points: Source points (Nx2 numpy array).
    :param dst_points: Destination points (Nx2 numpy array).
    :param iterations: Number of iterations for RANSAC.
    :param threshold: Threshold to consider a point as an inlier.
    :return: Best homography matrix found, inlier mask.
    """
    best_homography = None
    max_inliers = 0
    n_points = src_points.shape[0]

    for _ in range(iterations):
        # Randomly select 4 points for homography computation
        indices = np.random.choice(n_points, 4, replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]

        # Estimate homography using these points
        H = create_homography_matrix(src_sample, dst_sample)

        # Project src_points using the estimated homography
        ones = np.ones((src_points.shape[0], 1))
        homogeneous_src_points = np.hstack([src_points, ones])
        projected_points = (H @ homogeneous_src_points.T).T
        projected_points = projected_points[:, :2] / projected_points[:, [2]]

        # Calculate distances between projected and actual destination points
        distances = np.sqrt(np.sum((projected_points - dst_points) ** 2, axis=1))

        # Count inliers
        inliers_count = np.sum(distances < threshold)

        # Update the best homography matrix if more inliers are found
        if inliers_count > max_inliers:
            best_homography = H
            max_inliers = inliers_count

    # Re-estimate homography using all inliers if a model was found
    if best_homography is not None:
        inliers = np.sqrt(np.sum((projected_points - dst_points) ** 2, axis=1)) < threshold
        best_homography = create_homography_matrix(src_points[inliers], dst_points[inliers])

    return best_homography, inliers
    
def homographies_from_corresponding_points(pts_in_map_from_config, pts_in_frame_from_config):
    homographies = []
    for i in range(len(pts_in_map_from_config)):
        pts_in_map = np.array(pts_in_map_from_config[i][1:], dtype=float)
        pts_in_map = pts_in_map.reshape(int(len(pts_in_map)/2), 2)
        pts_in_frame = np.array(pts_in_frame_from_config[i][1:], dtype=float)
        pts_in_frame = pts_in_frame.reshape(int(len(pts_in_frame)/2), 2)
        homography = [0, int(pts_in_frame_from_config[i][0])]
        homography.extend(create_homography_matrix(pts_in_map, pts_in_frame).flatten())
        homographies.append(homography)
    return homographies

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("It's necessary the config path")
        sys.exit(1)
    config_path = sys.argv[1]
    config = parse_config_file(config_path)

    if config['transforms'][0][0] == 'homography':
        if config['transforms'][0][1] == 'all':
            features = io.loadmat(config['keypoints_out'])['features']
            frames_to_process = features.shape[1]
            homographies = homographies_from_features(features, frames_to_process)
        elif config['transforms'][0][1] == 'map':
            if len(config['pts_in_map']) != len(config['pts_in_frame']):
                print("Different amount of pts_in_map and pts_in_frame defined inside the config file")
                sys.exit(1)
            homographies = homographies_from_corresponding_points(config['pts_in_map'], config['pts_in_frame'])
        data={'transforms': np.array(homographies).transpose()}
        io.savemat(config['transforms_out'], data)
    else:
        print("The only acceptable type is \"homography\"")