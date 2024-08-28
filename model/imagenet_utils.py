import tensorflow as tf
from tensorflow.keras import backend

def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Args:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.

    Returns:
      A tuple of two integers specifying the padding for the height and width.
    """
    img_dim = 2 if backend.image_data_format() == "channels_first" else 1
    input_size = inputs.shape[img_dim : (img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    if input_size[0] is None:
        # For dynamic input size, we assume padding of 0 as placeholder
        return (0, 0)
    
    input_height, input_width = input_size
    kernel_height, kernel_width = kernel_size

    if (input_height % 2 == 0 and input_width % 2 == 0):
        # If the input dimensions are even, padding needed to keep output dimension same
        pad_along_height = max(kernel_height - (input_height % kernel_height), 0)
        pad_along_width = max(kernel_width - (input_width % kernel_width), 0)
    else:
        # If input dimensions are odd, adjust to make sure padding keeps the dimensions
        pad_along_height = max(kernel_height - (input_height % kernel_height), 0)
        pad_along_width = max(kernel_width - (input_width % kernel_width), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return ((pad_top, pad_bottom), (pad_left, pad_right))
