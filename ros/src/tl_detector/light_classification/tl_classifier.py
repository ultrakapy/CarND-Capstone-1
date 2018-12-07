import os
import numpy as np
import visualization_utils as vis_util
import label_map_util
import tensorflow as tf
import time
import cv2

from graph_utils import load_graph
from PIL import ImageDraw, Image
from styx_msgs.msg import TrafficLight


SAVE_MONITOR_IMAGE = False # to save monitor image for debug.

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        sess, _ = load_graph('models/sim_model.pb')

        self.sess = sess
        self.sess_graph = self.sess.graph
        # Definite input and output Tensors for sess
        self.image_tensor = self.sess_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.sess_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.sess_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.sess_graph.get_tensor_by_name('detection_classes:0')
        self.num_classes = 3

        self.label_map = label_map_util.load_labelmap("./labelmap.pbtxt")
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)


    def get_classification(self, image, wp = 0):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        cv2_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)        
        image_np_expanded = np.expand_dims(cv2_image, axis=0)

        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: image_np_expanded})

        prediction = 4
        min_score_thresh=.6
        sq_boxes = np.squeeze(boxes)
        sq_classes = np.squeeze(classes).astype(np.int32)
        sq_scores = np.squeeze(scores)

        for i in range(sq_boxes.shape[0]):
            if sq_scores is None or sq_scores[i] > min_score_thresh:
                if sq_classes[i] in self.category_index.keys():
                    prediction = sq_classes[i]
                    print("find traffic light:%s  color:%s   pred_score:%s"%(prediction, str(self.category_index[sq_classes[i]]['name']), sq_scores[i]))
                    min_score_thresh = sq_scores[i] 



        if SAVE_MONITOR_IMAGE: #or wp > 1900:

            dt_str = str(time.time()).replace('.','')
            file_name = "Monitor_IMG_" + dt_str + '_'+ str(int(prediction - 1)) +'.jpg'
            output_filename = "light_classification/IMGS/tl_origin/" + file_name
            cv2.imwrite(output_filename, image)

            vis_image = vis_util.visualize_boxes_and_labels_on_image_array(
                     image,
                     #np.squeeze(boxes),
                     #np.squeeze(classes).astype(np.int32),
                     #np.squeeze(scores),
                     sq_boxes,
                     sq_classes,
                     sq_scores,
                     self.category_index,
                     use_normalized_coordinates=True,
                     line_thickness=1
                     #min_score_thresh = min_score_thresh
                     )
            image_pil = Image.fromarray(cv2.cvtColor(vis_image,cv2.COLOR_BGR2RGB))
            file_name = "Monitor_IMG_" + dt_str + '_'+ str(int(prediction - 1)) +'.png'
            output_filename = "light_classification/IMGS/tl_predict_output/" + file_name
            image_pil.save(output_filename, 'PNG')

        rtn = TrafficLight.UNKNOWN

        if prediction == 1:
            rtn = TrafficLight.RED
        elif prediction == 2:
            rtn = TrafficLight.YELLOW
        elif prediction == 3:
            rtn = TrafficLight.GREEN

        return rtn
