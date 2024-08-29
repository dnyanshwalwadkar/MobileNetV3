import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Reshape, Multiply, Add, UpSampling2D
from mobilenetV3 import MobileNetV3Small

def LiteR_ASPP(features):
    # 1x1 Conv, BN, ReLU
    x1 = Conv2D(128, (1, 1), padding='same')(features)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    # Global Average Pooling, followed by 1x1 Conv and Sigmoid activation
    x2 = GlobalAveragePooling2D()(features)
    x2 = Reshape((1, 1, -1))(x2)
    x2 = Conv2D(128, (1, 1), padding='same')(x2)
    x2 = tf.keras.activations.sigmoid(x2)
    x2 = UpSampling2D(size=(49, 49), interpolation='bilinear')(x2)  # Upsample to match feature map size

    # Multiply the two paths
    x = Multiply()([x1, x2])

    # Bilinear upsample
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    # 1x1 Conv to reduce to 19 channels (for 19 classes)
    x = Conv2D(19, (1, 1), padding='same')(x)

    return x

def SegmentationHead(backbone_output):
    # Lite R-ASPP Head
    features = LiteR_ASPP(backbone_output)

    # Final 1x1 Conv to match output channels
    x = Conv2D(19, (1, 1), padding='same')(features)

    # Bilinear upsampling to match the input image size
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    return x

def build_segmentation_model(backbone):
    backbone_output = backbone.output  # Extract feature map from backbone

    segmentation_output = SegmentationHead(backbone_output)

    # Create the model
    model = tf.keras.models.Model(inputs=backbone.input, outputs=segmentation_output)

    return model

# Example Usage with MobileNetV3 Small or Large
# Assume `mobilenet_v3_small` and `mobilenet_v3_large` are defined MobileNetV3 models.

# For Small Model
backbone_small = MobileNetV3Small(input_shape=(224, 224, 3), alpha=1.0, include_top=False)
model_small = build_segmentation_model(backbone_small)
print(model_small.summary())
# # For Large Model
# backbone_large = mobilenet_v3_large(input_shape=(224, 224, 3), alpha=1.0, include_top=False)
# model_large = build_segmentation_model(backbone_large)

