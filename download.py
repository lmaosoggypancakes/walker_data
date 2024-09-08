from roboflow import Roboflow
rf = Roboflow(api_key="P5y0tIh3N4FIBqK740W1")
project = rf.workspace("mrl-zse1m").project("muadib")
version = project.version(1)
dataset = version.download("yolov8")
                