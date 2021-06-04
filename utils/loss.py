import tensorflow as tf
from tensorflow.keras.losses import Loss


class BoxLoss(Loss):
    def __init__(self, delta):
        super(BoxLoss, self).__init__(reduction='none', name='BoxLoss')
        self.delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = tf.square(difference)
        loss = tf.where(tf.less(absolute_difference, self.delta),
                        0.5 * squared_difference, absolute_difference - 0.5)
        return tf.reduce_sum(loss, axis=-1)

class ClassLoss(Loss):
    def __init__(self, alpha, gamma):
        super(ClassLoss, self).__init__(reduction='none', name='ClassLoss')
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self.alpha, (1.0 - self.alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self.gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

class FocalLoss(Loss):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, delta=1.0):
        super(FocalLoss, self).__init__(reduction='auto', name='FocalLoss')
        self.clf_loss = ClassLoss(alpha, gamma)
        self.box_loss = BoxLoss(delta)
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(tf.cast(y_true[:, :, 4], dtype=tf.int32), 
                                depth=self.num_classes, dtype=tf.float32)
        cls_predictions = y_pred[:, :, 4:]

        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)        
        cls_loss = self.clf_loss(cls_labels, cls_predictions)       
        box_loss = self.box_loss(box_labels, box_predictions)
        cls_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, cls_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        cls_loss = tf.math.divide_no_nan(tf.reduce_sum(cls_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = cls_loss + box_loss
        return loss     