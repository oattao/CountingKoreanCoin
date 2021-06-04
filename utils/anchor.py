import tensorflow as tf

class AnchorBox:
    def __init__(self, aspect_ratios=[0.5, 1.0, 2.0], 
                 scales=[0, 1/3, 2/3],
                 stride_range=[3, 4, 5, 6, 7],
                 areas = [32, 64, 128, 256, 512]):
        self.aspect_ratios = aspect_ratios
        self.scales = [2 ** x for x in scales]
        self.stride_range = stride_range
        self.strides = [2 ** x for x in self.stride_range]
        self.areas = [x ** 2 for x in areas]
        self.anchor_dims = self.compute_dims()
        self.num_anchors = len(self.aspect_ratios) * len(self.scales)

    def compute_dims(self):
        anchor_dims_all = []
        for area in self.areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                h = tf.math.sqrt(area / ratio)
                w = area / h
                dims = tf.reshape(tf.stack([w, h], axis=-1), [1, 1, 2])
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self.strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self.num_anchors, 1])
        dims = tf.tile(self.anchor_dims[level - 3], [feature_height, feature_width, 1, 1])
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(anchors, [feature_height * feature_width * self.num_anchors, 4])

    def get_anchors(self, image_height, image_width):
        anchors = [self._get_anchors(tf.math.ceil(image_height / 2 ** i),
                                     tf.math.ceil(image_width / 2 ** i), i)
                    for i in self.stride_range]
        return tf.concat(anchors, axis=0)