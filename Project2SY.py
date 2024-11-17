# Importing Necessary Libraries:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Check TensorFlow and Keras Versions
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}\n")

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# STEP 1 - DATA PROCESSING
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

# Define image dimensions and batch size
img_height = 500
img_width = 500
batch_size = 32
img_channel = 3
img_shape = (img_width, img_height, img_channel)

# Define Directory Paths for Training and Validation Data
train_directory = 'DataSet/train'  # Relative path to train folder
valid_directory = 'DataSet/valid'  # Relative path to validation folder

# Data Augmentation for Training Data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values between 0 and 1
    shear_range=0.2,  # Apply shear transformations
    zoom_range=0.2,    # Apply random zoom within range
    horizontal_flip = True)

# Data augmentation for validation data (only rescaling)
valid_datagen = ImageDataGenerator(
    rescale=1./255)  # Only rescale validation images, no augmentation

# Data Generators for Train and Validation
train_generator = train_datagen.flow_from_directory (
    train_directory,
    image_size=(img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical')

validation_generator = valid_datagen.flow_from_directory (
    valid_directory,
    image_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode ='categorical')

print("\nTrain generator:", train_generator.cardinality().numpy(), "samples")
print("Validation generator:", validation_generator.cardinality().numpy(), "samples")
print("Class indices:", train_generator.class_names)

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# STEP 2 - NEURAL NETWORK ARCHITECTURE DESIGN
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

# Model Setup
dcnn_model = models.Sequential()

# Layer 1: Convolution with 32 filters and MaxPooling
dcnn_model.add(layers.Input(shape=img_shape))  # Using the previously defined image shape
dcnn_model.add(layers.Conv2D(32, (3, 3), 
                             activation='relu'))
dcnn_model.add(layers.MaxPooling2D((2, 2)))

# Layer 2: Convolution with 64 filters and LeakyReLU Activation

dcnn_model.add(layers.Conv2D(64, (3, 3)))
dcnn_model.add(LeakyReLU(negative_slope=0.01))  # LeakyReLU for introducing non-linearity
dcnn_model.add(layers.MaxPooling2D((2, 2)))
dcnn_model.add(layers.Dropout(0.2))  # Dropout to reduce overfitting

# Layer 3: Convolution with 128 filters and MaxPooling
dcnn_model.add(layers.Conv2D(128, (3, 3), 
                             activation='relu'))
dcnn_model.add(layers.MaxPooling2D((2, 2)))
dcnn_model.add(layers.Dropout(0.35))  # Dropout to reduce overfitting

# Layer 4: Convolution with 256 filters and MaxPooling
dcnn_model.add(layers.Conv2D(256, (3, 3), 
                             activation='relu'))
dcnn_model.add(layers.MaxPooling2D((2, 2)))
dcnn_model.add(layers.Dropout(0.4))  # Dropout to reduce overfitting

dcnn_model.add(layers.Flatten())

# Output Dense Layer for multi-class classification (3 classes)
dcnn_model.add(layers.Dense(3, activation='softmax'))  # Output layer with 3 classes (softmax)

# Dense Layers
dcnn_model.add(layers.Dense(64, 
                            activation='relu'))  # Fully connected layer
dcnn_model.add(layers.Dropout(0.55))  # Dropout for regularization






