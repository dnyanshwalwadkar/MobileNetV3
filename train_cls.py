import os
import json
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
import math
def create_custom_model(cfg):
    """Create and return a custom MobileNetV3 model based on configuration."""
    shape = (int(cfg['height']), int(cfg['width']), 3)
    n_class = int(cfg['class_number'])
    
    if cfg['model'] == 'large' and cfg['Block'] == 'seb':
        from model.mobilenetV3 import MobileNetV3Large
        model = MobileNetV3Large(input_shape=shape, classes=2,alpha=0.15,include_top=True)
        
    elif cfg['model'] == 'small' and cfg['Block'] == 'seb':
        from model.mobilenetV3 import MobileNetV3Small
        model = MobileNetV3Small(input_shape=shape, classes=2,alpha=0.15,include_top=True)
        
    elif cfg['model'] == 'small'and cfg['Block'] == 'AGB':
        from model.mobilenetv3_withAGB import MobileNetV3Small
        model = MobileNetV3Small(input_shape=shape, classes=2,alpha=0.50,include_top=True)
        
    elif cfg['model'] == 'large'and cfg['Block'] == 'AGB':
        from model.mobilenetv3_withAGB import MobileNetV3Large
        model = MobileNetV3Large(input_shape=shape, classes=2,alpha=0.15,include_top=True)
        
    else:
        raise ValueError(f"Unknown model type: {cfg['model']}")
    
    return model

def display_model_summaries():
    """Create and display model summaries for custom and prebuilt MobileNetV3 models."""
    
    # Load the configuration
    with open('config/config.json', 'r') as f:
        cfg = json.load(f)

    print("Custom MobileNetV3 Model Summary:")
    custom_model = create_custom_model(cfg)
    custom_model.summary()

    # print("\nPrebuilt MobileNetV3 Large Model Summary:")
    # prebuilt_large = MobileNetV3Large(input_shape=(int(cfg['height']), int(cfg['width']), 3), alpha=0.15, include_top=False, weights=None)
    # prebuilt_large.summary()

    # print("\nPrebuilt MobileNetV3 Small Model Summary:")
    # prebuilt_small = MobileNetV3Small(input_shape=(int(cfg['height']), int(cfg['width']), 3),classes=2, include_top=False,alpha=0.15, weights=None)
    # prebuilt_small.summary()

if __name__ == '__main__':
    display_model_summaries()
