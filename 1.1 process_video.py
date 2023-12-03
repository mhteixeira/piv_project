import sys
import cv2 
import numpy as np
from scipy import io

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

def get_features(video_path):
    vid_capture = cv2.VideoCapture(video_path)
    frame_count = vid_capture.get(7)
    features = np.zeros((1, int(frame_count)), dtype=object)

    current_frame = 0 
    while(vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool 
        # and the second is frame
        ret, frame = vid_capture.read()
            
        sift = cv2.SIFT_create()
        if ret == True:
            # getting keypoints and descriptor
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            keypoints, descriptor = sift.detectAndCompute(frame_gray, None)

            # getting the location of each keypoint
            x_location = []
            y_location = []
            for keypoint in keypoints:
                x_location.append(keypoint.pt[0])
                y_location.append(keypoint.pt[1])
            
            concatenation = np.insert(np.transpose(descriptor), [0, 1], [x_location, y_location], axis=0)
            
            features[0, current_frame] = concatenation

            current_frame += 1
            
        else:
            break
    vid_capture.release()
    return features

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("It's necessary the config path")
        sys.exit(1)
    config_path = sys.argv[1]
    config = parse_config_file(config_path)
    features = get_features(config['videos'])
    data={'features': features}
    io.savemat(config['keypoints_out'], data)