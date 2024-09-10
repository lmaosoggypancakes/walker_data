from ultralytics import YOLO
from roboflow import Roboflow
import sys
import datetime
import cv2
import random
import math
from util import crop, _filter

WIN_NAME = "smallgatu lmaooo"
RED_TO_RED = 144.222*2 # 150px/10cm
PX_TO_CM = RED_TO_RED / 10
SAMPLING = 1 # 1/SAMPING of frames will be used
if __name__ == "__main__":
    vels = []
    last_point = None
    if len(sys.argv) != 2:
        print(sys.argv)
        print(f"usage: {sys.argv[0]} path/to/video")
        sys.exit()
    # Load a model
    outfn = "out/turn/" + sys.argv[1].split("/")[-1].split(".")[0] + ".csv"
    print(outfn)
    out = open(outfn, "w")
    out.write("t,x_px,y_px,x_cm,y_cm\n")
    out.close()
    out = open(outfn, "a")
    print("loading model...")
    model = YOLO("weights.pt")
    print("loading video...")
    cap = cv2.VideoCapture(sys.argv[1])
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frames)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    # calculate duration of the video 
    print("reading frames...")
    seconds = round(frames / fps, 3)
    frame_length = round(1 / fps, 3)
    t=0
    for frame in crop(cap):
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        boxes = results[0].boxes.xyxy
        cv2.imshow(WIN_NAME, cv2.resize(annotated_frame, (0,0), fx=0.5, fy=0.5) )
        if len(boxes) == 1:
            cx = (boxes[0][0] + boxes[0][2]) / 2
            cy = (boxes[0][1] + boxes[0][3]) / 2
            center = (round(cx.item(), 2), round(cy.item(), 2))
            if last_point:
                vx = (cx - last_point[0])/frame_length
                vy = (cy - last_point[1])/frame_length
            else: 
                vx = vy = 0
            v = round(math.sqrt(vx ** 2 + vy ** 2), 10) / SAMPLING
            vels.append(v)
            v_cm = round(v / PX_TO_CM, 10)
            out.write(f"{t}, {center[0]}, {center[1]}, {center[0]/PX_TO_CM}, {center[1]/PX_TO_CM}\n")
            last_point = center
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            while True:
                if cv2.waitKey(1) & 0xFF == ord(" "):
                    break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        t +=  frame_length
    vels_smooth = _filter(vels, 20)
    with open("filtered.csv", "w") as f:
        f.write("t,v\n ")
        for i in range(0, len(vels_smooth)):
            t = i*frame_length
            f.write(f"{t}, {vels_smooth[i]}\n")

    out.close()
    print("file saved to "  + outfn)
    print("data points: " + str(int(frames)))
    cap.release()
    cv2.destroyAllWindows()
