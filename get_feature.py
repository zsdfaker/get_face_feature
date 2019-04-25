# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:27:10 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import facenet
import glob
import pickle
from scipy.spatial import distance
image_size = 200 #don't need equal to real image size, but this value should not small than this
modeldir = './model/20170512-110547.pb' #change to your model dir


print('建立facenet embedding模型')
tf.Graph().as_default()
sess = tf.Session()

facenet.load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
feature = []
print('facenet embedding模型建立完毕')

scaled_reshape = []
for image in glob.glob("C:/Users/Administrator/my_deeplearning_work/face_recognition/data/IMFDB_final_crop/ValidationData/*.jpg"):
    print(image[-9:])
    inputs = scipy.misc.imread(image, mode='RGB')
    inputs = cv2.resize(inputs, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    inputs = facenet.prewhiten(inputs)
    scaled_reshape.append(inputs.reshape(-1,image_size,image_size,3))
    emb_array = np.zeros((1, embedding_size))
    emb_array[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False })[0]
    feature.append(emb_array)

#dist = distance.euclidean(feature[0], feature[1])
#
#print("128维特征向量的欧氏距离：%f "%dist)
def save_feature(feature, outputfile = 'my_feature.pkl'):
    with open(outputfile, 'wb') as f:
        pickle.dump(feature, f)
    return
save_feature(feature)