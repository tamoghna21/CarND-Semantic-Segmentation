from moviepy.editor import VideoFileClip

import scipy.misc
import numpy as np


import tensorflow as tf
import sys

# Third-party imports
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from sklearn.metrices import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import imageio
import cv2

#from tqdm import tqdm

def genn_output_image(sess, predicts, keep_prob, image_pl, input_image, image_shape):
    #print("entered here")
    image = scipy.misc.imresize(input_image, image_shape)
    image = image.reshape(1,*image.shape)#Adds a dimension at the beginning
    im_softmax = sess.run([predicts],feed_dict={image_pl: image,keep_prob: 1.0})
    
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    #print(np.shape(mask))
    #print(np.shape(image[0]))
    street_im = scipy.misc.toimage(image[0])
    street_im.paste(mask, box=None, mask=mask)
    
    return np.array(street_im)


class MyVideoProcessor(object):
    def __init__(self,sess,image_pl,keep_prob,predicted):
        self.sess = sess
        self.predicted=predicted
        self.keep_prob=keep_prob
        self.image_pl=image_pl
        self.image_shape = (160, 576)
        return
    def pipeline(self, rgb_frame):
        op_frame = genn_output_image(self.sess, self.predicted, self.keep_prob, self.image_pl, rgb_frame, self.image_shape)
        return op_frame
        
        
loaded_graph = tf.Graph()
load_dir = './model.ckpt'

with tf.Session(graph=loaded_graph) as sess:
    #Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)
    print("Model loaded")
    
    # Gather required tensor references
    input_image = loaded_graph.get_tensor_by_name('image_input:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    predicted = loaded_graph.get_tensor_by_name('predicts:0')
    
    my_video_processor_object = MyVideoProcessor(sess,input_image,keep_prob,predicted)
    project_video_output = 'test_video_output.mp4'
    clip = VideoFileClip("test_video.mp4")
    #clip = VideoFileClip("project_video.mp4").subclip(0,4)
    

    white_clip = clip.fl_image(my_video_processor_object.pipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(project_video_output, audio=False)
print("Done")