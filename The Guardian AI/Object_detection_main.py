import os

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import numpy
from utils.color_recognition_module import color_recognition_api

#CarModelRecognition
from dataset import torch, os, LocalDataset, transforms, np, get_class, num_classes, preprocessing2, Image, m, s
from config import *

from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models import resnet, vgg

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from matplotlib import pyplot as plt
from numpy import unravel_index
import gc
import argparse




flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/2_Trim.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', './outputs/2.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', True, 'count objects being tracked on screen')
flags.DEFINE_boolean('roi', True, 'Draw ROI and count')

def main(_argv):
    # Definition of the parameters  (???????????????)
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort  (?????????????????????)
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric  (????????????????????????)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker  (??????????????????)
    tracker = Tracker(metric)

    # load configuration for object detector  (??????????????????????????????)
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set  (?????????????????????????????????tflite??????)
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model   (??????????????????tensorflow????????????)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture  (??????????????????)
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set  (???????????????????????????????????????????????????)
    if FLAGS.output:
        # by default VideoCapture returns float instead of int   (??????????????????VideoCapture??????float?????????int)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0

    # ???????????????????????????????????????
    object_dic = {}

    # while video is running   (???????????????)
    while True:
        return_value, frame1 = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set   (??????????????????????????????tflite??????)
        if FLAGS.framework == 'tflite':

            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set   (?????????????????????????????????yolov3????????????)
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements   (??????????????????numpy????????????????????????????????????)
        num_objects = valid_detections.numpy()[0]  # ??????????????????????????????  int???
        bboxes = boxes.numpy()[0]
        bboxes1 = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]  # ??????????????????index,??????np

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        #  (??????????????????????????????ymin, xmin, ymax, xmax???> xmin, ymin??????????????????)
        original_h, original_w, _ = frame.shape

        bboxes= utils.format_boxes(bboxes1, original_h, original_w)


        # store all predictions in one parameter for simplicity when calling functions
        # (?????????????????????????????????????????????????????????????????????????????????)
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config  (??????????????????????????????)
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file (??????????????????.names?????????????????????)
        #allowed_classes = list(class_names.values())  # ????????????????????????

        # custom allowed classes (uncomment line below to customize tracker for only people)
        # (?????????????????????(???????????????????????????????????????????????????))
        allowed_classes = ['person', 'dog']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        # (???????????????????????????????????????????????????????????????allowed_classes???????????????)

        # ????????????????????????????????????????????????????????????dict={'name':num} ????????????????????????????????????
        dict = {}
        for i in allowed_classes:
            class_indx = (list(class_names.keys()))[list(class_names.values()).index(i)]
            class_num = np.count_nonzero(classes == class_indx)
            dict[i] = class_num

        # ????????????????????????????????????????????????????????????????????????????????????????????????
        names = []
        deleted_indx = []
        for i in range(num_objects):
            # ?????????index
            class_indx = int(classes[i])
            # ????????????
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        # ??????????????????????????????
        count = len(names)
        # ?????????????????????
        if FLAGS.count:
            # ???????????????
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (0, 255, 0), 2)
            y = 70
            # ??????????????????
            for key, value in dict.items():
                if value != 0:
                    cv2.putText(frame, "{} being tracked: {}".format(key, value), (5, y),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                    y += 35

            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes  (???????????????allowed_classes?????????)
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker  (??????yolo???????????????????????????)
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map  (?????????????????????)
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression  (??????non-maxima??????)
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker  (???????????????)
        tracker.predict()
        tracker.update(detections)

        # ???????????????????????????????????????ROI??????????????????
        target = []

        track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (0, 255, 255), (255, 0, 255), (255, 127, 255),
                        (127, 0, 255), (127, 0, 127)]

        # update tracks  (????????????)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()



            imgSelect = frame1[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            #print("box", bbox)
            #print("imgselect:", imgSelect, imgSelect is None, imgSelect==[], imgSelect.all(), imgSelect.shape)
            if imgSelect.shape[0] == 0 or imgSelect.shape[1] == 0 or imgSelect.shape[2] == 0:
                continue


            predicted_color = color_recognition_api.color_recognition(imgSelect)
            #print(predicted_color)
            #predicted_CarModel = test_sample(imgSelect)
            #print(predicted_CarModel)
            #detected_vehicle_image = frame[int(bbox[1]):int(bbox[1]) + int(bbox[3]),
                                     #int(bbox[0]):int(bbox[0]) + int(bbox[2])]
            #predicted_color = color_recognition_api.color_recognition(detected_vehicle_image)
            #print(predicted_color)

            # draw bbox on screen  (??????????????????bbox)
            # ????????????0-????????????x,1-????????????y???2-????????????x???3-????????????y
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[0]) + (len(class_name) + len(str(predicted_color))) * 17, int(bbox[1] + 20)), color,
                          -1)
            cv2.putText(frame, class_name + "-" + str(predicted_color) ,
                        (int(bbox[0]), int(bbox[1] + 10)), 0, 0.75,
                        (255, 255, 255), 2)

            # if enable info flag then print details about each track (??????????????????????????????????????????????????????????????????)
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
                                                                                                    class_name, (
                                                                                                    int(bbox[0]),
                                                                                                    int(bbox[1]),
                                                                                                    int(bbox[2]),
                                                                                                    int(bbox[3]))))

            # ?????????????????????
            center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2), int(bbox[2] - bbox[0]),
                      int(bbox[3] - bbox[1])]

            # ????????????????????????
            # ????????????????????????ID????????????{'id':{'trace':[[],[],[],??????[]],'trace_frames':num},'id':{'trace':[[],[],[],??????[]]}}

            if not "%d" % track.track_id in object_dic:
                # ????????????id????????????key(ID):val{????????????????????????}   ??????????????????10?????????????????????
                object_dic["%d" % track.track_id] = {"trace": [], 'traced_frames': 10}
                object_dic["%d" % track.track_id]["trace"].append(center)
                object_dic["%d" % track.track_id]["traced_frames"] += 1

            # ????????????????????????
            else:
                object_dic["%d" % track.track_id]["trace"].append(center)
                object_dic["%d" % track.track_id]["traced_frames"] += 1

            # ??????????????????roi????????????????????????
            if FLAGS.roi:
                # ????????????roi?????????   [173, 456], [966, 91], [1240, 122], [574, 515]
                pts1 = np.array([[(931, 209), (1268, 227), (1229, 712), (351, 520)]], np.int32)
                pts1 = pts1.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts1], True, (0, 255, 255), thickness=2)

                # ?????????????????????roi?????????
                # ??????????????????????????????????????????
                x = int((bbox[0] + bbox[2]) / 2)
                y = int(bbox[3]) - 10  # ????????????
                # ?????????4??????????????????Lad\Lbc\Lab\Lcd,a????????????b????????????c????????????d????????????
                yab = round(-0.46 * x + 535, 2)
                ybc = round(0.11 * x + 100.29, 2)
                ycd = round(-0.1 * x + 800.710, 2)
                yda = round(0.15 * x + 430.55, 2)
                # ???????????????????????????roi???
                if (y > yab and y > ybc and y < ycd and y < yda):
                    target.append(x)
                    cv2.putText(frame, str('enter'), (int(bbox[2] - 65), int(bbox[3] - 5)),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 1)
        cv2.putText(frame, "ROI count: {}".format(str(len(target))), (1500, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                    (255, 0, 0), 2)

        # ????????????
        for s in object_dic:
            i = int(s)

            # ????????????????????????????????????????????????????????????????????????????????????????????????
            # xlist, ylist, wlist, hlist = [], [], [], []

            # ????????????????????????
            if len(object_dic["%d" % i]["trace"]) > 20:
                for k in range(len(object_dic["%d" % i]["trace"]) - 20):
                    del object_dic["%d" % i]["trace"][k]

            # # # ????????????
            if len(object_dic["%d" % i]["trace"]) > 2:
                for j in range(1, len(object_dic["%d" % i]["trace"]) - 1):
                    pot1_x = object_dic["%d" % i]["trace"][j][0]
                    pot1_y = object_dic["%d" % i]["trace"][j][1]
                    pot2_x = object_dic["%d" % i]["trace"][j + 1][0]
                    pot2_y = object_dic["%d" % i]["trace"][j + 1][1]
                    # if pot2_x == pot1_x and pot1_y == pot2_y:
                    #     del object_dic["%d" % i]
                    clr = i % 9  # ??????????????????
                    cv2.line(frame, (pot1_x, pot1_y), (pot2_x, pot2_y), track_colors[clr], 2)

        # ????????????????????????????????????
        for s in object_dic:
            if object_dic["%d" % int(s)]["traced_frames"] > 0:
                object_dic["%d" % int(s)]["traced_frames"] -= 1
        for n in list(object_dic):
            if object_dic["%d" % int(n)]["traced_frames"] == 0:
                del object_dic["%d" % int(n)]

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file  (????????????????????????????????????????????????)
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
