# import libraries
from ultralytics import YOLO
import cv2
import util
from util import get_car, read_license_plate, write_csv

from sort.sort import *
obj_tracker = Sort()

import matplotlib
matplotlib.use('TKAgg')

results = {}

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO(r"D:\alpd_code\models\licence_plate_detector.pt")

# load video
video = cv2.VideoCapture(r"D:\alpd_code\sample2.mp4")
vehicles = [2, 3, 5, 7] # car, motorbike, bus, truck

# read 
ret = True
frame_number = -1
while ret:
    frame_number += 1
    ret, frame = video.read()
    if ret:
        results[frame_number] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        obj_detections = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                obj_detections.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = obj_tracker.update(np.asarray(obj_detections))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]       
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            x1car, y1car, x2car, y2car, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                cropped_license_plate = frame[int(y1):int(y2), int(x1):int(x2), :]

                # process license plate
                cropped_license_plate_gray = cv2.cvtColor(cropped_license_plate, cv2.COLOR_BGR2GRAY)
                _, cropped_license_plate_thresh = cv2.threshold(cropped_license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(cropped_license_plate_thresh)
                if license_plate_text is not None:
                    results[frame_number][car_id] = {'car': {'bbox': [x1car, y1car, x2car, y2car]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
write_csv(results, r"D:\alpd_code\test\test2.csv")