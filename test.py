from ultralytics import YOLO
from roboflow import Roboflow
import sys
import datetime
import cv2
import math
from crop import crop

WIN_NAME = "smallgatu lmaooo"
RED_TO_RED = 15 # 48pix/10cm

if __name__ == "__main__":
    init = None
    final = None
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} path/to/video")
        sys.exit()
    # Load a model
    print("loading model...")
    model = YOLO("weights.pt")
    print("loading video...")
    cap = cv2.VideoCapture(sys.argv[1])
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = cap.get(cv2.CAP_PROP_FPS) 
    
    # calculate duration of the video 
    f=0
    print("reading frames...")
    seconds = round(frames / fps, 3)
    for frame in crop(cap):
        f+=1
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        boxes = results[0].boxes.xyxy
        # cv2.imshow(WIN_NAME, annotated_frame)
        if len(boxes) != 0:
            cx = (boxes[0][0] + boxes[0][2]) / 2
            cy = (boxes[0][1] + boxes[0][3]) / 2
            center = (round(cx.item(), 2), round(cy.item(), 2))
        # print(center)
        # Display the annotated frame
            if not init:
                init = center
            if f > frames-30:
                final = center
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    print(init, final)
    dist = round(math.dist(init, final), 2)
    dist_cm = dist / RED_TO_RED
    print("=== statistics ===")
    print(f"video dimensions: {annotated_frame.shape[:2]}")
    print(f"initial starting point: {init}")
    print(f"final point: {final}")
    print(f"distance travelled (px): {dist}")
    print(f"distance travelled (cm): {dist_cm}")
    print(f"SPEED: {dist_cm/seconds} cm/s")

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()