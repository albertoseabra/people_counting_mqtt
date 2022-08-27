from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib import config
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest

import os
import threading

import paho.mqtt.client as mqtt

os.environ["MQTT_BROKER"] = "test.mosquitto.org"
os.environ["MQTT_TOPIC"] = "alberto/num_persons"

# camera orientation, if vertical it considers entering the direction up to down, to false entering is left to right
vertical_orientation = True

# if debug will show video feed with extra information
debug = True

people_inside = 0

t0 = time.time()


def run_mqtt():
    global people_inside
    # Start by connecting to mqtt
    client = mqtt.Client()
    client.connect(os.environ.get("MQTT_BROKER", "test.mosquitto.org"), int(os.environ.get("MQTT_PORT", 1883)))
    print("connected successfully to mqtt")
    while True:
        print("Total people inside: {}".format(people_inside))
        client.publish(topic=os.environ.get("MQTT_TOPIC", "alberto/testing"),
                           payload=str(people_inside), qos=1, retain=True)
        time.sleep(5)

def run_main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    # confidence default 0.4
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=20,
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())

    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # if a video path was not supplied, grab a reference to the ip camera
    if not args.get("input", False):
        print("[INFO] Starting the live stream..")
        vs = VideoStream(config.url).start()
        time.sleep(2.0)

    # otherwise, grab a reference to the video file
    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])

    # initialize the frame dimensions (we'll set them as soon as we read the first frame from the video)
    W = None
    H = None

    # instantiate centroid tracker, list to store dlib trackers, a dictionary to map object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    total_in = 0
    total_out = 0
    global people_inside
    people_inside = 0

    # start the frames per second throughput estimator
    fps = FPS().start()

    # if config.Thread:
    #     vs = thread.ThreadingClass(config.url)

    # loop over frames from the video stream
    while True:
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        # if we are viewing a video and we did not grab a frame then we have reached the end of the video
        if args["input"] is not None and frame is None:
            break

        # resize the frame to have a maximum width of 500 pixels then convert the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=900)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % args["skip_frames"] == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                if confidence > args["confidence"]:
                    # extract the index of the class label from the detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        if debug:
            # draw a line in the center to determine whether they were entering or exiting
            if vertical_orientation:
                cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
            else:
                cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 0, 0), 3)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current object ID and create one if not
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)

            else:
                # the difference between the y-coordinate of the *current* centroid and the mean of *previous* centroids
                # will tell us in which direction the object is moving (negative for 'up' and positive for 'down')
                if vertical_orientation:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)
                    # check to see if the object has been counted or not
                    if not to.counted:
                        # if the direction is negative  AND the centroid is above the center line, count the object
                        if direction < 0 and centroid[1] < H // 2:
                            total_out += 1
                            to.counted = True
                        elif direction > 0 and centroid[1] > H // 2:
                            total_in += 1
                            to.counted = True

                else:
                    # positive/entering direction from left to right
                    x = [c[0] for c in to.centroids]
                    direction = centroid[0] - np.mean(x)
                    to.centroids.append(centroid)
                    if not to.counted:
                        if direction < 0 and centroid[0] < W // 2:
                            total_out += 1
                            to.counted = True
                        elif direction > 0 and centroid[0] > W // 2:
                            total_in += 1
                            to.counted = True

                # compute the sum of total people inside
                people_inside = total_in - total_out
                if people_inside < 0:
                    people_inside = 0
                    total_in = total_in + (total_out - total_in)

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the object on the output frame
            if debug:
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # construct a tuple of information we will be displaying on the
        info = [
            ("Exit", total_out),
            ("Enter", total_in),
            ("Status", status),
        ]

        info2 = [
            ("Total people inside", people_inside),
        ]

        if debug:
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            for (i, (k, v)) in enumerate(info2):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # show the output frame
            cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # Initiate a simple log to save data at end of the day
        if config.Log:
            datetimee = [datetime.datetime.now()]
            d = [datetimee, people_inside]
            export_data = zip_longest(*d, fillvalue='')

            with open('Log.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(("End Time", "In", "Out", "Total Inside"))
                wr.writerows(export_data)

        # increment the total number of frames processed thus far and then update the FPS counter
        totalFrames += 1
        fps.update()

        if config.Timer:
            # Automatic timer to stop the live stream. Set to 8 hours (28800s).
            t1 = time.time()
            num_seconds = (t1 - t0)
            if num_seconds > 28800:
                break

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # # if we are not using a video file, stop the camera video stream
    # if not args.get("input", False):
    # 	vs.stop()
    #
    # # otherwise, release the video file pointer
    # else:
    # 	vs.release()

    # issue 15
    # if config.Thread:
    #     vs.release()

    # close any open windows
    cv2.destroyAllWindows()


if config.Scheduler:
    ##Runs for every 1 second
    # schedule.every(1).seconds.do(run)
    ##Runs at every day (09:00 am). You can change it.
    schedule.every().day.at("09:00").do(run_main)

    while 1:
        schedule.run_pending()

else:
    trd1 = threading.Thread(target=run_mqtt, daemon=True)
    trd2 = threading.Thread(target=run_main)

    trd1.start()
    trd2.start()
