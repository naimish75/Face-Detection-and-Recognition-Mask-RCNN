import cv2
import time
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

from worker import Worker
from enums import Actions, Requests

import os
import imutils
import numpy as np
import tensorflow as tf
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import utils
import visualize
from imutils.video import WebcamVideoStream
import random
import final


def trap_exc_during_debug(*args):
    # when app raises uncaught exception, print info
    print(args)


# Print the error without crashing the application
sys.excepthook = trap_exc_during_debug


class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class Window(QWidget):
    controller = pyqtSignal(Requests)

    def __init__(self):
        super().__init__()

        self.setGeometry(100, 100, 800, 600)

        self.thread = Worker(self)

        # Connect toggle callback (start/pause buttons)
        self.controller.connect(self.thread.handleRequest)

        # Connect action reporter
        self.thread.communicator.connect(self.action)

        # Connect frame changed callback
        self.thread.frameChanged.connect(self.frameChanged)

        grid = QGridLayout()
        grid.setSpacing(5)

        # Label displaying status
        self.statusLabel = QLabel('Status: Please choose a video')
        grid.addWidget(self.statusLabel, 0, 0, 1, 8)

        # Label displaying fps
        self.fpsLabel = QLabel('')
        self.fpsLabel.setAlignment(Qt.AlignRight)
        grid.addWidget(self.fpsLabel, 0, 8, 1, 4)

        # Add horizontal separator
        grid.addWidget(QHLine(), 1, 0, 1, 12)

        # Button to open live detection
        self.realtimeButton = QPushButton('Realtime Detection')
        self.realtimeButton.clicked.connect(self.realTime)
        self.realtimeButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.realtimeButton, 2, 0, 1, 12)

        # Button to open file picker
        self.chooseVideoButton = QPushButton('Choose Video')
        self.chooseVideoButton.clicked.connect(self.chooseVideo)
        self.chooseVideoButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.chooseVideoButton, 3, 0, 1, 12)

        # Button that loads the weights, opens video window
        self.startButton = QPushButton('Start')
        self.startButton.clicked.connect(self.start)
        self.startButton.setEnabled(False)
        self.startButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.startButton, 4, 0, 2, 4)

        # Button that pauses video playback
        self.pauseButton = QPushButton('Pause')
        self.pauseButton.clicked.connect(self.pause)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.pauseButton, 4, 4, 1, 4)

        # Button that resumes video playback
        self.resumeButton = QPushButton('Resume')
        self.resumeButton.clicked.connect(self.resume)
        self.resumeButton.setEnabled(False)
        self.resumeButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.resumeButton, 5, 4, 1, 4)

        # Button stops video playback and closes the window
        self.stopButton = QPushButton('Stop')
        self.stopButton.clicked.connect(self.stop)
        self.stopButton.setEnabled(False)
        self.stopButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.stopButton, 4, 8, 2, 4)

        # Label displaying the file path of the chosen video
        self.filePathLabel = QLabel('File: None')
        self.filePathLabel.setWordWrap(True)
        grid.addWidget(self.filePathLabel, 6, 0, 1, 12)

        # Label displaying the current time of the video
        self.currentTimeLabel = QLabel('0:00:00')
        grid.addWidget(self.currentTimeLabel, 7, 0, 1, 1)

        # Slider to scrub through the video
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.NoTicks)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.setTime)
        self.slider.setEnabled(False)
        grid.addWidget(self.slider, 7, 1, 1, 10)

        # Label displaying the duration of the video
        self.finalTimeLabel = QLabel('0:00:00')
        grid.addWidget(self.finalTimeLabel, 7, 11, 1, 1)

        # Add horizontal separator
        grid.addWidget(QHLine(), 8, 0, 1, 12)

        # Button to toggle the mask rcnn detection
        self.detectToggle = QCheckBox('Detect objects')
        self.detectToggle.setEnabled(False)
        self.detectToggle.stateChanged.connect(self.toggleDetection)
        grid.addWidget(self.detectToggle, 9, 0, 1, 12)

        # Checkbox to toggle displaying masks on the detected objects
        self.masksToggle = QCheckBox('Show masks')
        self.masksToggle.setEnabled(False)
        self.masksToggle.stateChanged.connect(self.toggleMasks)
        self.masksToggle.setChecked(True)
        grid.addWidget(self.masksToggle, 10, 0, 1, 6)

        # Checkbox to toggle displaying bounding boxes on the detected objects
        self.boxesToggle = QCheckBox('Show bounding boxes')
        self.boxesToggle.setEnabled(False)
        self.boxesToggle.stateChanged.connect(self.toggleBoxes)
        self.boxesToggle.setChecked(True)
        grid.addWidget(self.boxesToggle, 10, 6, 1, 6)

        # Add horizontal separator
        grid.addWidget(QHLine(), 11, 0, 1, 12)

        # Checkbox to toggle saving the video
        self.saveToggle = QCheckBox('Save video')
        self.saveToggle.setEnabled(False)
        self.saveToggle.stateChanged.connect(self.toggleSave)
        self.saveToggle.setChecked(False)
        grid.addWidget(self.saveToggle, 12, 0, 1, 12)

        # Label for the video save path
        self.savePathLabel = QLabel('Location: None')
        self.savePathLabel.setWordWrap(True)
        grid.addWidget(self.savePathLabel, 13, 0, 1, 12)

        # Button Logout
        self.logout = QPushButton('Logout')
        self.logout.clicked.connect(self.Logout)
        self.logout.setEnabled(True)
        self.logout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.logout, 13, 12, 1, 2)

        self.setLayout(grid)


    def Logout(self):
        window.hide()
        os.system("python login.py")
        sys.exit(app.exec_())
        
        

    def toggleSave(self):
        if self.saveToggle.isChecked():
            self.controller.emit(Requests.SAVE_ON)
        else:
            self.controller.emit(Requests.SAVE_OFF)
            return

        # _ is the filter
        filePath = QFileDialog.getExistingDirectory(self, "Choose a directory...")

        self.savePathLabel.setText('Location: ' + filePath)

        self.thread.setSave(filePath)

    def setTime(self):
        seconds = self.slider.value()

        paused = self.thread.paused

        self.pause()
        time.sleep(0.1)
        fps = self.thread.capture.get(cv2.CAP_PROP_FPS)
        new_frame_number = round(seconds * fps)
        print("Changing to frame", new_frame_number)
        self.thread.capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame_number)

        # Only resume if it was already paused
        if not paused:
            self.resume()

    def toggleDetection(self):
        # Update the worker with the new detection setting
        if self.detectToggle.isChecked():
            self.controller.emit(Requests.DETECT_ON)
        else:
            self.controller.emit(Requests.DETECT_OFF)

        self.masksToggle.setEnabled(self.detectToggle.isChecked())
        self.boxesToggle.setEnabled(self.detectToggle.isChecked())

    def toggleMasks(self):
        # Update the worker with the new mask setting
        if self.masksToggle.isChecked():
            self.controller.emit(Requests.MASKS_ON)
        else:
            self.controller.emit(Requests.MASKS_OFF)

        if not self.masksToggle.isChecked() and not self.boxesToggle.isChecked():
            self.detectToggle.setChecked(False)

    def toggleBoxes(self):
        # Update the worker with the new boxes setting
        if self.boxesToggle.isChecked():
            self.controller.emit(Requests.BOXES_ON)
        else:
            self.controller.emit(Requests.BOXES_OFF)

        if not self.masksToggle.isChecked() and not self.boxesToggle.isChecked():
            self.detectToggle.setChecked(False)

    # Real Time Recognition Code Block
    def realTime(self, checked):

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

        def ImageUpdateSlot(self, Image):
            self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

        def CancelFeed(self):
            self.Worker1.stop()

        class Worker1(QThread):
            ImageUpdate = pyqtSignal(QImage)
            def run(self):
                self.ThreadActive = True
                Capture = cv2.VideoCapture(0)
                while self.ThreadActive:
                    ret, frame = Capture.read()
                    if ret:
                        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        FlippedImage = cv2.flip(Image, 1)
                        ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                        Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                        self.ImageUpdate.emit(Pic)
            def stop(self):
                self.ThreadActive = False
                self.quit()

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
        #config.display()

        DEVICE = "/gpu:0"

        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=TRAINED_MODEL_PATH, config=config)

        model.load_weights(TRAINED_MODEL_PATH, by_name=True)

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
        window.hide()
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
        window.show()



    def alert(self, s):
        err = QErrorMessage(self)
        err.showMessage(s)

    def chooseVideo(self):
        # _ is the filter
        filePath, _ = QFileDialog.getOpenFileName(self, 'Choose a video file', '', 'Videos Files | *.mp4;')

        if filePath is '':
            self.statusLabel.setText('Status: Please choose a video')
            self.startButton.setEnabled(False)
            return

        self.filePathLabel.setText('File: ' + filePath)

        # Set file path
        self.thread.setVideo(filePath)

        # GUI freezes and the text isn't rendered if done before loading paths
        if self.thread.loadedWeights:
            self.statusLabel.setText('Status: Ready')
        else:
            self.statusLabel.setText('Status: Loading weights...')

        # Loads weights and idles
        if not self.thread.isRunning():
            self.thread.start()

        self.saveToggle.setEnabled(True)

        if self.thread.loadedWeights:
            self.startButton.setEnabled(True)

    def frameChanged(self):
        formatted = round(self.thread.fps * 100) / 100
        self.fpsLabel.setText('FPS: ' + str(formatted))

        millis = self.thread.capture.get(cv2.CAP_PROP_POS_MSEC)
        self.currentTimeLabel.setText(self.formatTime(millis))

        # Block signals when we set the value to stop any callbacks from being triggered
        self.slider.blockSignals(True)
        self.slider.setValue(round(millis / 1000))
        self.slider.blockSignals(False)

        # Set the maximum label if it's not been set yet
        if self.finalTimeLabel.text() == '0:00:00':
            fps = self.thread.capture.get(cv2.CAP_PROP_FPS)
            total_frames = self.thread.capture.get(cv2.CAP_PROP_FRAME_COUNT)

            # Video duration in seconds
            duration = (float(total_frames) / float(fps))

            formatted = self.formatTime(duration * 1000)
            self.finalTimeLabel.setText(formatted)
            self.slider.setMaximum(duration)

    def formatTime(self, milliseconds):
        minutes, seconds = divmod(milliseconds / 1000, 60)
        hours, minutes = divmod(minutes, 60)
        return "%d:%02d:%02d" % (hours, minutes, seconds)

    def action(self, action):
        # Reported when weighs are about to load
        if action is Actions.LOADING_WEIGHTS:
            self.statusLabel.setText('Status: Loading weights...')

        # Reported when weights have been loaded
        if action is Actions.LOADED_WEIGHTS:
            self.statusLabel.setText('Status: Ready')
            self.startButton.setEnabled(True)
            self.pauseButton.setEnabled(False)
            self.resumeButton.setEnabled(False)
            self.stopButton.setEnabled(False)
            self.detectToggle.setEnabled(True)

        # Reported back when the video has been loaded
        if action is Actions.LOADED_VIDEO:
            self.statusLabel.setText('Status: Running...')
            self.startButton.setEnabled(False)
            self.resumeButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.stopButton.setEnabled(True)
            self.slider.setEnabled(True)

        # Reported back when the video playback has finished
        if action is Actions.FINISHED:
            self.statusLabel.setText('Status: Finished')
            self.stop()

    def start(self):
        self.startButton.setEnabled(False)
        self.resumeButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.stopButton.setEnabled(False)
        self.detectToggle.setEnabled(True)
        self.saveToggle.setEnabled(False)
        self.controller.emit(Requests.START)

    def pause(self):
        self.pauseButton.setEnabled(False)
        self.resumeButton.setEnabled(True)
        self.detectToggle.setEnabled(True)
        self.controller.emit(Requests.PAUSE)

    def resume(self):
        self.pauseButton.setEnabled(True)
        self.resumeButton.setEnabled(False)
        self.detectToggle.setEnabled(True)
        self.controller.emit(Requests.RESUME)

    def stop(self):
        self.startButton.setEnabled(True)
        self.resumeButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.stopButton.setEnabled(False)
        self.detectToggle.setEnabled(False)
        self.slider.setEnabled(False)
        self.saveToggle.setEnabled(True)
        self.slider.setValue(0)
        self.currentTimeLabel.setText('0:00:00')
        self.finalTimeLabel.setText('0:00:00')
        self.fpsLabel.setText('')

        self.controller.emit(Requests.STOP)


class RealTime(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Detection")


app = QApplication(sys.argv)
window = Window()
window.setWindowTitle("God's Eye")
window.show()
sys.exit(app.exec_())

