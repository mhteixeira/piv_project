import cv2

cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
num = 0

while cap.isOpened():

    succes, img = cap.read()
    k = cv2.waitKey(3)

    if k == 30:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('frames/img' + str(num) + '.png', img)
        print("image has been saved")
        num += 1
    cv2.imshow('webcam', img)

cap.release()
cv2.destroyAllWindows()