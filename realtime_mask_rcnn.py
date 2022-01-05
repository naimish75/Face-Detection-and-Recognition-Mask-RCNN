import os
import sys
import tensorflow as tf
import cv2
import time
import mrcnn.model as modellib
from mrcnn import utils
import visualize
import imutils
import final

ROOT_DIR = os.path.abspath("./")
sys.path.append(os.path.join(ROOT_DIR))

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_PATH = os.path.join(MODEL_DIR, "models\\mask_rcnn_object_0025.h5")

config = final.CustomConfig()


class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

DEVICE = "/gpu:0"
TEST_MODE = "inference"

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(MODEL_PATH, by_name=True)

class_names = ["Rahul Jasani", "Rahul Patel", "Nilesh"]

stream = cv2.VideoCapture(0)

while True:
    grabbed, frame = stream.read()
    if not grabbed:
        break

    results = model.detect([frame], verbose=1)
    r = results[0]

    boxes = r["rois"]
    masks = r["masks"]
    class_ids = r["class_ids"]
    scores = r["scores"]

    start = time.time()
    masked_image = visualize.get_masked_image(frame, boxes, masks, class_ids, class_names, scores)

    end = time.time()
    print("Inference time: {:.2f}s".format(end - start))

    cv2.imshow("", masked_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


stream.release()
cv2.destroyAllWindows()