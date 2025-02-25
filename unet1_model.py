import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm

def unet(input_size=(256, 256, 3), pretrained_weights=None):  # ✅ Change input_size to (256,256,3)
    model = sm.Unet(
        'resnet34', 
        encoder_weights='imagenet', 
        input_shape=input_size,  # ✅ Fix input shape
        classes=1,               
        activation='sigmoid'      # ✅ Ensure binary segmentation output
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model
