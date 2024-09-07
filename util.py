import cv2
import sys
import os
import math
import numpy as np

def crop(vc):
    while vc.isOpened():
        success, frame = vc.read()
        if not success:
            break
        frame = cv2.resize(frame, (1080,1920))
        cropped = frame[0:1920,0:1080]
        yield cropped

def _filter(data, n):
    """
    running mean average of window size n
    """
    i = 0
    av = []
    while i < len(data) - n+1:
        window = data[i: i+n]
        average = sum(window) / n
        av.append(average)
        i+=1
    return av