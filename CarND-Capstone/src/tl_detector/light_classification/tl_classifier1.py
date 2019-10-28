

import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import os
import time
from glob import glob
from styx_msgs.msg import TrafficLight
import cv2
from numpy import zeros, newaxis
import rospkg





class TLClassifier(object):
    def __init__(self):

        self.signal_classes = ['Red', 'Green', 'Yellow']

        self.signal_status = None
        
        self.tl_box = None
        
        
        ros_root = rospkg.get_ros_root()

        r = rospkg.RosPack()
        
        path = r.get_path('tl_detector')

        self.cls_model = load_model(path + '/model_f_lr-4_ep140_ba32.h5') 
        
        #'ssd_mobilenet_v1_coco_11_06_2017'
        
        PATH_TO_CKPT = path + '/frozen_inference_graph.pb'

        self.detection_graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:

              serialized_graph = fid.read()
              od_graph_def.ParseFromString(serialized_graph)
              tf.import_graph_def(od_graph_def, name='')
            
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')


def get_classification(self, image):
    
    (im_width, im_height) = image.size
    img_full_np =  np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    img_full_np_copy = np.copy(img_full_np)
    
    
    category_index={1: {'id': 1, 'name': u'person'},
                        2: {'id': 2, 'name': u'bicycle'},
                        3: {'id': 3, 'name': u'car'},
                        4: {'id': 4, 'name': u'motorcycle'},
                        5: {'id': 5, 'name': u'airplane'},
                        6: {'id': 6, 'name': u'bus'},
                        7: {'id': 7, 'name': u'train'},
                        8: {'id': 8, 'name': u'truck'},
                        9: {'id': 9, 'name': u'boat'},
                        10: {'id': 10, 'name': u'traffic light'},
                        11: {'id': 11, 'name': u'fire hydrant'},
                        13: {'id': 13, 'name': u'stop sign'},
                        14: {'id': 14, 'name': u'parking meter'}}  
    
    
    with self.detection_graph.as_default():
          image_expanded = np.expand_dims(img_full_np, axis=0)
          (boxes, scores, classes, num_detections) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections], feed_dict={self.image_tensor: image_expanded})

          boxes=np.squeeze(boxes)
          classes =np.squeeze(classes)
          scores = np.squeeze(scores)

          cls = classes.tolist()

          # Find the first occurence of traffic light detection id=10
          idx = next((i for i, v in enumerate(cls) if v == 10.), None)
          # If there is no detection
          if (idx == None) or (scores[idx] <= 0.02):
              return TrafficLight.UNKNOWN
          else:
            
              #*************corner cases***********************************
              dim = img_full_np.shape[0:2]
              boxes = boxes[idx]
              
              height, width = dim[0], dim[1]
              box_pixel = [int(boxes[0]*height), int(boxes[1]*width), int(boxes[2]*height), int(boxes[3]*width)]
              box =  np.array(box_pixel) 
      
              box_h = box[2] - box[0]
              box_w = box[3] - box[1]
              ratio = box_h/(box_w + 0.01)
              # if the box is too small, 20 pixels for simulator
              if (box_h <10) or (box_w<10) or (ratio< 1.5):
                  return TrafficLight.UNKNOWN
              
              else:    
                       
                  self.tl_box = box
                  img_resize = cv2.resize(img_full_np_copy[box[0]:box[2], box[1]:box[3]], (32, 32))
                  img_resize=cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB) 
                  img_resize = np.expand_dims(img_resize, axis=0).astype('float32')
                  img_resize/=255.
                  predict = self.cls_model.predict(img_resize)
                  predict = np.squeeze(predict, axis =0)
                  tl_color = self.signal_classes[np.argmax(predict)]
                  
                  state_idx = np.argmax(predict)
                  
                  
                  if state_idx == 0:
                    return TrafficLight.RED
                  elif state_idx == 1:
                    return TrafficLight.GREEN
                  elif state_idx == 2:
                    return TrafficLight.YELLOW

    
    