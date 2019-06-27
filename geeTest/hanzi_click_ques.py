#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os


# In[2]:


import numpy as np
from keras import layers
from keras.layers import GRU, Permute, LSTM,TimeDistributed, Lambda, Bidirectional, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.utils import layer_utils, Sequence
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import string
import os
import tensorflow as tf
import keras.layers as KL

import keras.backend as K
K.set_image_data_format('channels_last')


# In[3]:


class CaptchaModel:
    pass


# In[4]:


characters = ''
class Config():
    def __init__(self, config={}):
        for k in config.keys():
            self.__setattr__(k, config[k])
_config = {'width': 116, 'height': 40, 'char_num': 6, 'lower': True, 'characters': characters, 'channels': 1, 'train_vc_file_name_handle': lambda s:"%s.%s" % (''.join([(c if c in characters[:-1] else '@') for c in s[s.rfind('_') + 1:].split('.')[0] if re.match('[\u4e00-\u9fa5@]', c)]), 'jpg')}
_config = Config(_config)


# In[5]:

import base64
import sys

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers as KL
from PIL import Image
import io

import queue
import threading
import time


class PredictResultHolder:

    def __init__(self):
        self.result = None
        self.event = threading.Event()

    def get(self, timeout=5):
        if timeout < 0:
            timeout = 5
        self.event.wait(timeout)
        return self.result

    def set(self, result):
        self.result = result
        self.event.set()
        self.event.clear()


class Predicter:

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                _, self.model = CaptchaModel(_config).create_model(0)
                self.model.load_weights('geeTest/hanzi_click_ques.hdf5')
               
        self.ctc_decoder = None
        self.predict_queue = queue.Queue()
        self.predict_queue_monitor = threading.Event()
        self.handling = True
        self.__predict_queue_consume()

    def predict_captcha(self, captcha_image):
        holder = PredictResultHolder()
        self.predict_queue.put((captcha_image, holder))
        self.predict_queue_monitor.set()
        self.predict_queue_monitor.clear()
        return holder.get()

    def __predict_queue_consume(self):
        def __consumer():
            with self.graph.as_default(), self.sess.as_default():
                ctc_decodes = {}
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
                    
                    buf = io.BytesIO()
                    img.save(buf, format='jpeg')

                    img = Image.open(buf)
                    img = np.array(img)
                    origin_shape = img.shape
                    if len(origin_shape) == 2:
                        assert _config.channels == 1
                        img = np.reshape(img, [_config.height, _config.width, 1])
                    if _config.channels == 1:
                        if len(origin_shape) > 2:
                            img = np.mean(img, -1)
                            img = img.reshape([_config.height, _config.width, 1])
                    img = np.reshape(img, [-1, _config.height, _config.width, 1]) / 255.
                    y_pred = self.model.predict(img)

                    if not self.ctc_decoder:
                        shape = y_pred.shape
                        inputs = K.placeholder(shape)
                        input_length = KL.Input(batch_shape=[None], dtype='int32')
                        ctc_decode = K.ctc_decode(inputs, input_length=input_length, greedy=True)[0][0]
                        self.ctc_decoder = K.function([inputs, input_length], [ctc_decode])
                    out = self.ctc_decoder([y_pred, np.ones(shape[0]) * shape[1]])[0][0][:_config.char_num]
                    predict_tuple[1].set(''.join([_config.characters[x] for x in out if x > -1]))

        th = threading.Thread(target=__consumer)
        th.start()

    def close(self):
        self.handling = False
        K.clear_session()



