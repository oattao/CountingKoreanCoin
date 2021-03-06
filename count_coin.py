import os 
import time
import pathlib
import argparse
import cv2 as cv
import numpy as np
from PIL import Image
import tensorflow as tf
from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils
from config.config import money_value


def load_image_to_numpy_array(path):
    return np.array(Image.open(path))

def compute_money(detections, min_score_thresh=.7):
    money = 0
    for mclass, score in zip(detections['detection_classes'], detections['detection_scores']):
        if score > min_score_thresh:
            money += money_value[mclass]
    return money

parser = argparse.ArgumentParser()
parser.add_argument('--image_path')    
parser.add_argument('--model_path')
args = parser.parse_args()
image_path = args.image_path
model_path = args.model_path

# Prepare environment
tf.get_logger().setLevel('ERROR')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Prepare label
path_to_labels = './config/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

# Load trained model
# path_to_model = 'models/exported_models/ssd_v3/saved_model'

print('Loading model...')
detect_fn = tf.saved_model.load(model_path)
print('Model loaded.')

image_np = load_image_to_numpy_array(image_path)
    
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]

detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
image_np_with_detections = image_np.copy()
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=20,
      min_score_thresh=.8,
      agnostic_mode=False)

money = compute_money(detections, min_score_thresh=.8)
cv.putText(image_np_with_detections, f'Total: {money} won', (0, 70),
           cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

image = Image.fromarray(image_np_with_detections)
image.save('data/toshow/output1.jpg')
image.show()