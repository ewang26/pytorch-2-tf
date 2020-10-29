import tensorflow as tf

def create_tf_layer_from_name(output, attributes, weight):
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
    elif attributes['name'] == 'Linear':
        in_features = attributes['in_features']
        out_features = attributes['out_features']
        bias = attributes['bias']

        with tf.variable_scope('linear'):
            weight = tf.get_variable(
                name='weight', shape=[in_features, out_features],
                initializer=weight)

            #TODO actually support bias
            output = tf.matmul(_input, weight)
            if use_bias:
                init_key = '%s/bias' % tf.get_variable_scope().name
                initializer = param_initializer.get(init_key, tf.constant_initializer([0.0] * out_units))
                bias = tf.get_variable(
                    name='bias', shape=[out_units],
                    initializer=initializer
                )
                output = output + bias
        return output
