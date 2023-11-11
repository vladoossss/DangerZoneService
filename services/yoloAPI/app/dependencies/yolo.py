from app.core.yolov5.detect_video import YoloBase

def get_yolo_instance():
    if YoloBase.instance:
        return YoloBase.instance
    return YoloBase()
