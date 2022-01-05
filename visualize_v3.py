import os
import sys
import cv2
import time
import imutils
import numpy as np
import tensorflow as tf
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import utils
import visualize
from imutils.video import WebcamVideoStream
import random
import final

ROOT_DIR = os.path.abspath("./")
sys.path.append(os.path.join(ROOT_DIR))
SAVE_PATH = os.path.join(ROOT_DIR, "MASKED_IMG")

MODEL_DIR = os.path.join(ROOT_DIR, "logs/models")

TRAINED_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_object_0065.h5")
config = final.CustomConfig()


class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

DEVICE = "/cpu:0"

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=TRAINED_MODEL_PATH, config=config)

model.load_weights(TRAINED_MODEL_PATH, by_name=True) #, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]

class_names = ["BG", "Rahul Patel", "Nilesh"]

cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("Could not open video camera")

colors = visualize.random_colors(len(class_names))
gentle_grey = (45, 64, 79)
white = (255, 255, 255)

OPTIMIZE_CAM = False
SHOW_FPS = False
SHOW_FPS_WO_COUNTER = True
PROCESS_IMG = True

if OPTIMIZE_CAM:
    vs = WebcamVideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(0)

if SHOW_FPS:
    fps_caption = "FPS: 0"
    fps_counter = 0
    start_time = time()

SCREEN_NAME = "Real-Time Recognition"
cv2.namedWindow(SCREEN_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(SCREEN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    if OPTIMIZE_CAM:
        frame = vs.read()
    else:
        grabbed, frame = vs.read()
        if not grabbed:
            break
        if SHOW_FPS_WO_COUNTER:
            start_time = time.time()

        if PROCESS_IMG:
            results = model.detect([frame])
            r = results[0]
            masked_image = visualize.display_instances_10fps(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], colors=colors, real_time=True)
            #cv2.imwrite(masked_image, SAVE_PATH)

        if PROCESS_IMG:
            s = masked_image
        else:
            s = frame

        width = s.shape[1]
        height = s.shape[0]
        top_left_corner = (width-120, height-20)
        bott_right_corner = (width, height)
        top_left_corner_cvtext = (width-80, height-5)

        if SHOW_FPS:
            fps_counter += 1
            if (time.time() - start_time) > 5:  # every 5 seconds
                fps_caption = "FPS: {:.0f}".format(fps_counter / (time.time() - start_time))

                fps_counter = 0
                start_time = time.time()
            ret, baseline = cv2.getTextSize(fps_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(s, (width - ret[0], height - ret[1] - baseline), bott_right_corner, gentle_grey, -1)
            cv2.putText(s, fps_caption, (width - ret[0], height - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1,
                        lineType=cv2.LINE_AA)

        if SHOW_FPS_WO_COUNTER:
            fps_caption = "FPS: {:.0f}".format(1.0 / (time.time() - start_time))
            ret, baseline = cv2.getTextSize(fps_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(s, (width - ret[0], height - ret[1] - baseline), bott_right_corner, gentle_grey, -1)
            cv2.putText(s, fps_caption, (width - ret[0], height - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1,
                        lineType=cv2.LINE_AA)

        s = cv2.resize(s, (1720, 1080))
        cv2.imshow(SCREEN_NAME, s)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if OPTIMIZE_CAM:
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()