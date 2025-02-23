import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam

def unet_model(input_size=(256, 256, 3), pretrained_weights=None):
    """
    Load a pretrained U-Net model with a ResNet34 encoder.
    """

    # Load pretrained U-Net with ResNet34 as the encoder
    model = sm.Unet('resnet34', encoder_weights='imagenet', input_shape=input_size)

    # Compile model with binary crossentropy loss
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Load pretrained weights if provided
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
