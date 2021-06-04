import os
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from utils.anchor import AnchorBox
from utils.image import (swap_xy, convert_to_xywh, convert_to_corners, 
                       compute_iou, random_flip_horizontal, resize_and_pad_image)


def preprocess_data(sample):
    image = sample['image']
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample['objects']['label'], dtype=tf.int32)
    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)
    bbox = tf.stack([bbox[:, 0] * image_shape[1],
                     bbox[:, 1] * image_shape[0],
                     bbox[:, 2] * image_shape[1],
                     bbox[:, 3] * image_shape[0]], axis=-1)
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id

def preprocess_data_from_tfrecord(sample, feature_description):
    parsed_sample = tf.io.parse_single_example(sample, feature_description)
    image = tf.image.decode_jpeg(parsed_sample['image/encoded'], channels=3)
    w = tf.cast(parsed_sample['image/width'], tf.float32)
    h = tf.cast(parsed_sample['image/height'], tf.float32)
    xmin = parsed_sample['image/object/bbox/xmin']
    ymin = parsed_sample['image/object/bbox/ymin']
    xmax = parsed_sample['image/object/bbox/xmax']
    ymax = parsed_sample['image/object/bbox/ymax']
    class_id = tf.cast(parsed_sample['image/object/class/label'], tf.int32)
    
    bbox = tf.stack([xmin, ymin, xmax, ymax], axis=1)

    image_shape = tf.shape(image)
    bbox = tf.stack([bbox[:, 0] / image_shape[1],
                     bbox[:, 1] / image_shape[0],
                     bbox[:, 2] / image_shape[1],
                     bbox[:, 3] / image_shape[0]], axis=-1)
    
    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)
    bbox = tf.stack([bbox[:, 0] * image_shape[1],
                     bbox[:, 1] * image_shape[0],
                     bbox[:, 2] * image_shape[1],
                     bbox[:, 3] * image_shape[0]], axis=-1)
    bbox = convert_to_xywh(bbox)

    return image, bbox, class_id


def preprocess_data_from_textline(line, image_path):
    parts = tf.strings.split(line, ' ')
    filepath = '{}{}'.format(image_path, os.path.sep).encode() + parts[0]
    raw_box = tf.map_fn(lambda x: tf.strings.split(x, ','), parts[1:])
    bbox_with_id = tf.strings.to_number(raw_box)
    # bbox_with_id = tf.cast(bbox_with_id, tf.int32)
    
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    
    bbox = bbox_with_id[:, :-1]
    class_id = tf.cast(bbox_with_id[:, -1], tf.int32)

    # normalize box
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

    bbox = tf.stack([bbox[:, 0] / image_shape[1],
                     bbox[:, 1] / image_shape[0],
                     bbox[:, 2] / image_shape[1],
                     bbox[:, 3] / image_shape[0]], axis=-1)

    image, bbox = random_flip_horizontal(image, bbox)
    # image = tf.cast(image, dtype=tf.float32)
    image, image_shape, _ = resize_and_pad_image(image)
    bbox = tf.stack([bbox[:, 0] * image_shape[1],
                     bbox[:, 1] * image_shape[0],
                     bbox[:, 2] * image_shape[1],
                     bbox[:, 3] * image_shape[0]], axis=-1)

    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


class LabelEncoder:
    def __init__(self, variance=[0.1, 0.1, 0.2, 0.2], dtype=tf.float32) :
        self.anchor_box = AnchorBox()
        self.box_variance = tf.convert_to_tensor(variance, dtype=tf.float32)

    def match_anchor_boxes(self, anchor_boxes, gt_boxes, 
                           match_iou=0.5, ignore_iou=0.4):
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))

        positive_mask = tf.cast(positive_mask, dtype=tf.float32)
        ignore_mask = tf.cast(ignore_mask, dtype=tf.float32)

        return matched_gt_idx, positive_mask, ignore_mask

    def compute_box_target(self, anchor_boxes, matched_gt_boxes):
        box_target = tf.concat([(matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:])], 
                                axis=-1)
        box_target /= self.box_variance
        return box_target

    def encode_sample(self, image_shape, gt_boxes, cls_ids):
        anchor_boxes = self.anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self.match_anchor_boxes(
            anchor_boxes, gt_boxes)
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self.compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids)
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        image_shape = tf.shape(batch_images)
        batch_size = image_shape[0]
        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self.encode_sample(image_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = preprocess_input(batch_images)
        return batch_images, labels.stack()