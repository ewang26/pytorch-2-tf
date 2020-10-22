import tensorflow as tf

def create_tf_layer_from_name(attributes, weight):
    '''
    From a dictionary of layer attributes, create a TensorFlow layer with the corresponding
    attributes and initialize its weight to a pretrained weight

    Args:
        attributes: a dictionary of layer attributes
        weight: a numpy array describing the weights of the layer

    Returns:
        tf_layer: a tf.nn layer matching the specifications
    '''

    if attributes['name'] == 'Conv2d':
        kernel_size = attributes['kernel_size']
        in_features = attributes['in_channels']
        out_features = attributes['out_channels']

        weight = tf.get_variable(
            name='weight',
            shape=[
                kernel_size,
                kernel_size,
                in_features,
            out_features],
            initializer=weight)

        output = tf.nn.conv2d(output, weight, [1, stride, stride, 1], 'VALID', data_format='NHWC')
    #TODO add more layers

