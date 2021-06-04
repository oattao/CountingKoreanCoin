import os
import io
import argparse
from PIL import Image
import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util
from config.config import label_dict

def create_tfrecord(data_path, annotation_file, output_file):
    class_dict = {value: key for key, value in label_dict.items()}

    tf_writer = tf.io.TFRecordWriter(output_file)

    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split(' ')
        filename = parts[0]
        boxes = parts[1:]

        with tf.io.gfile.GFile(os.path.join(data_path, filename), 'rb') as fid:
            encoded_jpg = fid.read()
        encode_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encode_jpg_io)
        w, h = image.size
        filename = filename.encode('utf8')
        image_format = b'jpg'
        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        classes_text = []
        labels = []
        for box in boxes:
            xmin, ymin, xmax, ymax, label = map(int, box.split(','))
            xmins.append(xmin/w)
            ymins.append(ymin/h)
            xmaxs.append(xmax/w)
            ymaxs.append(ymax/h)
            labels.append(label+1)
            classes_text.append(class_dict[label].encode('utf8'))
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(h),
            'image/width': dataset_util.int64_feature(w),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(labels),
            }))
        tf_writer.write(tf_example.SerializeToString())
    tf_writer.close()

if __name__ == '__main__':
    for name in ['train', 'val']:
        print(f'Creating {name} record...')
        create_tfrecord(data_path=f'./data/synthetic_images/{name}',
                        annotation_file=f'./data/synthetic_images/annotation_{name}.txt',
                        output_file=f'./data/synthetic_images/tf_{name}.record')
    print('Done.')





            
