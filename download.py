from roboflow import Roboflow
rf = Roboflow(api_key="P5y0tIh3N4FIBqK740W1")
project = rf.workspace("mrl-zse1m").project("walker_project")
version = project.version(2)
dataset = version.download("yolov8")