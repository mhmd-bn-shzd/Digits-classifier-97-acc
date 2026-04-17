import tensorflow as tf
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppresses info/warning logs

def residual_block(x, filters, kernel_size=3):
    """ResNet block with two conv layers and a skip connection."""
    shortcut = x
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def Classifier(input_shape=(28, 28, 1), num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    # x = layers.MaxPooling2D(2)(x) 
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs, outputs, name="MyResNet")
    return model

model = Classifier()
model.load_weights("Classifier.weights.h5")

img = cv2.imread("seven.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28,28))
img = img.astype('float32') / 255.0
img = img.reshape(1, 28, 28, 1)

prediction = model.predict(img)
predicted_digit = np.argmax(prediction)
confidence = np.max(prediction)

print(f"Predicted Digit: {predicted_digit}")
print(f"Confidence: {confidence * 100:.2f}%")

