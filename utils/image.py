import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def swap_xy(boxes):
    xmin, ymin, xmax, ymax = tf.split(boxes, 4, axis=-1)
    new_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    return new_boxes

def convert_to_xywh(boxes):
    cxcy = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    wh = boxes[..., 2:] - boxes[..., :2]
    new_boxes = tf.concat([cxcy, wh], axis=-1)
    return new_boxes

def convert_to_corners(boxes):
    xy_min = boxes[..., :2] - boxes[..., 2:] * 0.5
    xy_max = xy_min + boxes[..., 2:]    
    new_boxes = tf.concat([xy_min, xy_max], axis=-1)
    return new_boxes

def compute_iou(boxes1, boxes2):
    # boxes are in format [x, y, w, h]
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8)
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)   

def random_flip_horizontal(image, boxes):
    if tf.random.uniform(()) > 0.5:
        tf.image.flip_left_right(image)
        xmin, ymin, xmax, ymax = tf.split(boxes, 4, axis=-1)
        boxes = tf.concat([1 - xmax, ymin, 1 - xmin, ymax], axis=-1)
    return image, boxes

def resize_and_pad_image(image, min_side=800, max_side=1333,
                         jitter=[640, 1024], stride=128.0):
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape *= ratio
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(tf.math.ceil(image_shape / stride) * stride,
                                 dtype=tf.int32)
    image = tf.image.pad_to_bounding_box(image, 0, 0, padded_image_shape[0],
                                         padded_image_shape[1])
    return image, image_shape, ratio

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

def visualize_detections(image, boxes, classes, scores, figsize=(7, 7), 
                         linewidth=1, color=[0, 1, 0]):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=color, 
                               linewidth=linewidth)
        ax.add_patch(patch)
        ax.text(x1+5, y1-9, text, bbox={"facecolor": color, "alpha": 0.4},
                clip_box=ax.clipbox, clip_on=True)
    plt.show()
    return ax    