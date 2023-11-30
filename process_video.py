from setup import parse_config_file
import sys
import cv2 
import numpy as np
from scipy import io

def get_features(video_path):
    vid_capture = cv2.VideoCapture(video_path)
    frame_count = vid_capture.get(7)
    features = np.empty((1, int(frame_count)), dtype=object)

    current_frame = 0 
    while(vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool 
        # and the second is frame
        ret, frame = vid_capture.read()
        if ret == True:
            # getting keypoints and descriptor
            sift = cv2.SIFT_create()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            keypoints, descriptor = sift.detectAndCompute(frame_gray, None)

            # getting the location of each keypoint
            x_location = []
            y_location = []
            for keypoint in keypoints:
                x_location.append(keypoint.pt[0])
                y_location.append(keypoint.pt[1])
            ## (x, y, d)
            concatenation = np.insert(np.transpose(descriptor), [0, 1], [x_location, y_location], axis=0)
            #print(f'concatenation: {np.shape(concatenation)}') 
            features[0, current_frame] = concatenation

            current_frame += 1
        else:
            break

    vid_capture.release()
    cv2.destroyAllWindows()
    return features[0]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("It's necessary the config path")
        sys.exit(1)
    config_path = sys.argv[1]
    config = parse_config_file(config_path)
    features = get_features(config['videos'])
    data={'features': features}
    io.savemat(config['keypoints_out'], data)