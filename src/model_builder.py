from keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2

def create_turtle_model(num_classes):
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
    ])

    base_model = MobileNetV2(input_shape=(224, 224, 3), 
                             include_top=False, 
                             weights='imagenet')
    
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.RandomFlip("horizontal"),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model