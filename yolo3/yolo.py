import queue
import threading

import numpy as np

import colorsys
import os

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model
gpu_num=0

class PredictResultHolder:

    def __init__(self):
        self.result = None
        self.event = threading.Event()

    def get(self, timeout=15):
        if timeout < 0:
            timeout = 15
        self.event.wait(timeout)
        return self.result

    def set(self, result):
        self.result = result
        self.event.set()
        self.event.clear()


class YOLO(object):
    def __init__(self, model_path, anchors_path, classes_path, score=0.5, iou=0.5):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                self.model_path = model_path # model path or trained weights path
                self.anchors_path = anchors_path
                self.classes_path = classes_path
                self.score = score
                self.iou = iou
                self.sess = K.get_session()
                self.class_names = self._get_class()
                self.anchors = self._get_anchors()
                self.model_image_size = (416, 416) # fixed size or (None, None), hw
                self.boxes, self.scores, self.classes = self.generate()
                self.predict_queue = queue.Queue()
                self.predict_queue_monitor = threading.Event()
                self.handling = True
                self.__predict_queue_consume()

    def __predict_queue_consume(self):
        def __consumer():
            with self.graph.as_default(), self.sess.as_default():
                 while self.handling:
                     predict_tuple = self.predict_queue.get()
                     if not predict_tuple:
                         while self.predict_queue.empty():
                             try:
                                 self.predict_queue_monitor.wait(0.1)
                                 break
                             except Exception:
                                 pass
                         continue
                     img = predict_tuple[0]
                     predict_tuple[1].set(self.detect_image(img))

        th = threading.Thread(target=__consumer)
        th.start()

    def detect_image2(self, img):
        holder = PredictResultHolder()
        self.predict_queue.put((img, holder))
        self.predict_queue_monitor.set()
        self.predict_queue_monitor.clear()
        return holder.get()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False, by_name=True)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        results = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            results.append((predicted_class, score, (left, top), (right, bottom)))
        return results

    def close_session(self):
        self.sess.close()
        K.clear_session()
