import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from model import imagenet_utils


def relu(x):
    return layers.ReLU()(x)

def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)

def hard_swish(x):
    return layers.Activation("hard_swish")(x)

def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(
        keepdims=True, name=prefix + "squeeze_excite_avg_pool"
    )(inputs)
    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite_conv",
    )(x)
    x = layers.ReLU(name=prefix + "squeeze_excite_relu")(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite_conv_1",
    )(x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + "squeeze_excite_mul")([inputs, x])
    return x



def _inverted_res_block(
    x, expansion, filters, kernel_size, stride, se_ratio, activation, block_id
):
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    shortcut = x
    prefix = "expanded_conv_"
    infilters = x.shape[channel_axis]
    if block_id:
        prefix = f"expanded_conv_{block_id}_"
        x = layers.Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "expand_bn",
        )(x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            name=prefix + "depthwise_pad",
        )(x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="same" if stride == 1 else "valid",
        use_bias=False,
        name=prefix + "depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise_bn",
    )(x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=prefix + "project",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project_bn",
    )(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + "add")([shortcut, x])
    return x

def MobileNetV3(
    stack_fn,
    last_point_ch,
    input_shape=None,
    alpha=1.0,
    model_type="large",
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    # Check for the validity of the weights argument
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  "
            f"Received weights={weights}"
        )

    # Determine the input shape if not provided
    if input_shape is None and input_tensor is not None:
        if backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == "channels_first":
                rows = input_tensor.shape[2]
                cols = input_tensor.shape[3]
                input_shape = (3, cols, rows)
            else:
                rows = input_tensor.shape[1]
                cols = input_tensor.shape[2]
                input_shape = (cols, rows, 3)
    if input_shape is None and input_tensor is None:
        if backend.image_data_format() == "channels_last":
            input_shape = (None, None, 3)
        else:
            input_shape = (3, None, None)

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError(
            "Input size must be at least 32x32; Received `input_shape="
            f"{input_shape}`"
        )

    # Create the input layer
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    # Adjust the kernel and activation functions based on the model's complexity
    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25

    x = img_input

    # First convolutional layer with alpha scaling
    x = layers.Conv2D(
        _depth(16 * alpha),
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name="conv_bn",
    )(x)
    x = activation(x)

    # Stack the blocks of layers
    x = stack_fn(x, alpha=alpha)

    # Last convolutional layer with alpha scaling
    x = layers.Conv2D(
        _depth(last_point_ch * alpha),
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="last_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name="last_conv_bn",
    )(x)
    x = activation(x)

    x = layers.GlobalAveragePooling2D(keepdims=True, name="global_avg_pool")(x)
    x = layers.Reshape((_depth(last_point_ch * alpha),), name="reshape")(x)

    # Apply dropout if needed
    if dropout_rate:
        x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Add the top layer if include_top is set to True
    if include_top:
        x = layers.Dense(
            classes,
            activation=classifier_activation,
            name="predictions",
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Create the model
    model = Model(img_input, x, name="mobilenet_v3_" + model_type)

    return model

def _mobilenet_v3_small(x, alpha=1.0):
    x = _inverted_res_block(x, expansion=1, filters=_depth(16 * alpha), kernel_size=3, stride=2, se_ratio=0.25, activation=hard_swish, block_id=0)
    x = _inverted_res_block(x, expansion=72/16, filters=_depth(24 * alpha), kernel_size=3, stride=2, se_ratio=0.25, activation=hard_swish, block_id=1)
    x = _inverted_res_block(x, expansion=88/24, filters=_depth(24 * alpha), kernel_size=3, stride=1, se_ratio=0.25, activation=hard_swish, block_id=2)
    x = _inverted_res_block(x, expansion=4, filters=_depth(40 * alpha), kernel_size=5, stride=2, se_ratio=0.25, activation=hard_swish, block_id=3)
    x = _inverted_res_block(x, expansion=6, filters=_depth(40 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=4)
    x = _inverted_res_block(x, expansion=6, filters=_depth(40 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=5)
    x = _inverted_res_block(x, expansion=3, filters=_depth(48 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=6)
    x = _inverted_res_block(x, expansion=3, filters=_depth(48 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=7)
    x = _inverted_res_block(x, expansion=6, filters=_depth(96 * alpha), kernel_size=5, stride=2, se_ratio=0.25, activation=hard_swish, block_id=8)
    x = _inverted_res_block(x, expansion=6, filters=_depth(96 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=9)
    x = _inverted_res_block(x, expansion=6, filters=_depth(96 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=10)
    return x

def _mobilenet_v3_large(x, alpha=1.0):
    x = _inverted_res_block(x, expansion=1, filters=_depth(16 * alpha), kernel_size=3, stride=2, se_ratio=0.25, activation=hard_swish, block_id=0)
    x = _inverted_res_block(x, expansion=4, filters=_depth(24 * alpha), kernel_size=3, stride=2, se_ratio=0.25, activation=hard_swish, block_id=1)
    x = _inverted_res_block(x, expansion=3, filters=_depth(24 * alpha), kernel_size=3, stride=1, se_ratio=0.25, activation=hard_swish, block_id=2)
    x = _inverted_res_block(x, expansion=3, filters=_depth(40 * alpha), kernel_size=5, stride=2, se_ratio=0.25, activation=hard_swish, block_id=3)
    x = _inverted_res_block(x, expansion=3, filters=_depth(40 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=4)
    x = _inverted_res_block(x, expansion=3, filters=_depth(40 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=5)
    x = _inverted_res_block(x, expansion=6, filters=_depth(80 * alpha), kernel_size=5, stride=2, se_ratio=0.25, activation=hard_swish, block_id=6)
    x = _inverted_res_block(x, expansion=2.5, filters=_depth(80 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=7)
    x = _inverted_res_block(x, expansion=2.3, filters=_depth(80 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=8)
    x = _inverted_res_block(x, expansion=2.3, filters=_depth(80 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=9)
    x = _inverted_res_block(x, expansion=6, filters=_depth(112 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=10)
    x = _inverted_res_block(x, expansion=6, filters=_depth(112 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=11)
    x = _inverted_res_block(x, expansion=6, filters=_depth(160 * alpha), kernel_size=5, stride=2, se_ratio=0.25, activation=hard_swish, block_id=12)
    x = _inverted_res_block(x, expansion=6, filters=_depth(160 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=13)
    x = _inverted_res_block(x, expansion=6, filters=_depth(160 * alpha), kernel_size=5, stride=1, se_ratio=0.25, activation=hard_swish, block_id=14)
    return x


def MobileNetV3Small(input_shape=None, alpha=1.0, include_top=True, weights="imagenet", classes=2, **kwargs):
    return MobileNetV3(
        stack_fn=_mobilenet_v3_small,
        last_point_ch=1024,
        input_shape=input_shape,
        alpha=alpha,
        model_type="small",
        include_top=include_top,
        weights=weights,
        classes=classes,
        **kwargs
    )

def MobileNetV3Large(input_shape=None, alpha=1.0, include_top=True, weights="imagenet", classes=2, **kwargs):
    return MobileNetV3(
        stack_fn=_mobilenet_v3_large,
        last_point_ch=1280,
        input_shape=input_shape,
        alpha=alpha,
        model_type="large",
        include_top=include_top,
        weights=weights,
        classes=classes,
        **kwargs
    )