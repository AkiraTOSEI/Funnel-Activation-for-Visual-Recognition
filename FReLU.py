import tensorflow as tf


class FReLU(tf.keras.layers.Layer):
    def __init__(self,  **kwargs):
        super(FReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=1, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        # Be sure to call this somewhere!
        super(FReLU, self).build(input_shape)

    def call(self, x):
        h = self.conv(x)
        h = self.bn(h)
        out = tf.math.maximum(h, x)
        return out
