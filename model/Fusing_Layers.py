import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

def fuse_conv_bn(conv_layer, bn_layer):
    """
    Fuses the convolution and batch normalization layers into a single convolution layer.
    
    Args:
    - conv_layer: The convolution layer.
    - bn_layer: The batch normalization layer.
    
    Returns:
    - The fused convolution layer.
    """
    # Get the weights of the layers
    conv_weights = conv_layer.get_weights()
    bn_weights = bn_layer.get_weights()
    
    # Extract weights
    conv_kernel = conv_weights[0]  # Shape: (height, width, input_channels, output_channels)
    conv_bias = conv_weights[1] if len(conv_weights) > 1 else np.zeros(conv_kernel.shape[-1])
    bn_gamma, bn_beta, bn_mean, bn_var = bn_weights
    
    # Calculate the scale and offset
    bn_std = np.sqrt(bn_var + 1e-3)  # Add epsilon to variance
    scale = bn_gamma / bn_std
    offset = bn_beta - bn_mean * scale
    
    # Reshape scale and offset
    scale = scale.reshape(-1, 1, 1, 1)  # Shape: (output_channels, 1, 1, 1)
    offset = offset.reshape(-1)  # Shape: (output_channels,)
    
    # Broadcast scale and adjust kernel and bias
    fused_kernel = conv_kernel * scale
    if conv_bias is not None:
        fused_bias = conv_bias * scale.squeeze() + offset
    else:
        fused_bias = offset
    
    # Create a new Conv2D layer with fused weights and bias
    fused_conv = layers.Conv2D(
        filters=conv_layer.filters,
        kernel_size=conv_layer.kernel_size,
        strides=conv_layer.strides,
        padding=conv_layer.padding,
        use_bias=True,  # Set to True as we add bias manually
        kernel_initializer='zeros',
        bias_initializer='zeros'
    )
    
    # Set the new weights
    fused_conv.set_weights([fused_kernel, fused_bias])
    
    return fused_conv

def fuse_conv_bn_model(model):
    """
    Fuses all Conv2D and BatchNormalization layers in the model.
    
    Args:
    - model: The trained model with Conv2D and BatchNormalization layers.
    
    Returns:
    - The model with fused Conv2D layers.
    """
    # Create a copy of the model
    model_copy = tf.keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    
    # Iterate through the layers and fuse Conv2D and BatchNormalization layers
    new_layers = []
    for layer in model_copy.layers:
        if isinstance(layer, layers.Conv2D):
            # Find the corresponding BatchNormalization layer
            bn_layer = None
            for l in model_copy.layers:
                if isinstance(l, layers.BatchNormalization) and l.name.startswith(layer.name):
                    bn_layer = l
                    break
            if bn_layer:
                fused_conv = fuse_conv_bn(layer, bn_layer)
                new_layers.append(fused_conv)
            else:
                new_layers.append(layer)
        else:
            new_layers.append(layer)
    
    # Create a new model with the fused layers
    # Connect the layers manually
    x = model_copy.input
    for layer in new_layers:
        x = layer(x)
    
    new_model = tf.keras.models.Model(inputs=model_copy.input, outputs=x)
    
    return new_model


from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large

# Load the pre-trained MobileNetV3 model
# Replace `weights='imagenet'` with the path to weights if needed, or `None` for random initialization
model_small = MobileNetV3Small(weights='imagenet', include_top=True)
# Fuse the layers
model_small_fused = fuse_conv_bn_model(model_small)
print(model_small.summary())
print(model_small_fused.summary())