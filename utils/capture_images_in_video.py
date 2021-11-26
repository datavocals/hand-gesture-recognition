#!/usr/bin/python

import cv2

cap = cv2.VideoCapture("../samples/video/hand-gesture-video.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("../samples/video/hand-gesture-video.mp4")
    cv2.waitKey(1000)
    print("Wait for the header")

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
count = 0
while cap.isOpened():
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        cv2.imshow('video', frame)
        cv2.imwrite("../samples/pictures/frame-%d.jpg" % count, frame)
        count += 1
    else:
        break
cap.release()
    

