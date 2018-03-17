import cv2,time, numpy as np, os
from homography import run

vend = "http://172.20.40.228:4747/mjpegfeed?640x480"

cap = cv2.VideoCapture(vend)

cv2.namedWindow('Test',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Test', 600,350)

while(True):
    ret, frame = cap.read()
    if ret:
        results = run(frame)
        cv2.imshow('Test', results )
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
