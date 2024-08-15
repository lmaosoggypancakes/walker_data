import cv2
import sys
import os
def crop(vc):
    while vc.isOpened():
        success, frame = vc.read()
        if not success:
            break
        frame = cv2.resize(frame, (1080,1920))
        cropped = frame[0:1920,504:816]
        yield cropped

                