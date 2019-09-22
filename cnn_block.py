from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
#np.set_printoptions(threshold=np.nan)



def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


class CnnBlock:
	"""docstring for CnnBlock"""
	def __init__(self):
		self.model = faceRecoModel( input_shape=(3, 96, 96) )

	def load(self):
		self.model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
		load_weights_from_FaceNet(self.model)

	def encode_image(self, image):
		return img_to_encoding(image, self.model)

	def encoding_folder(image):
		dic = []
		path = "images/"+path
		print(path)
		l = os.listdir(path)
		for file in l:
			dic.append(self.encode_image(cv2.imread(path+file)))
		return dic



