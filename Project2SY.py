
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# ---------------------------------------
# ---------------------------------------
# STEP 2.1 - DATA PROCESSING (20 Marks)
# ---------------------------------------
# ---------------------------------------

# Define image dimensions and batch size
img_height = 500
img_width = 500
batch_size = 32

# Define Directory Paths for Training and Validation Data

train_directory = 'data/train'  # Relative path to train folder
valid_directory = 'data/valid'  # Relative path to validation folder
test_directory = 'data/test'    # Relative path to test folder

# ------------------------------------
# Data Augmentation for Training Data
# ------------------------------------

# Data augmentation for Training Data
train_datagen = ImageDataGenerator(
    rescale = 1.0/255,  # Normalize pixel values between 0 and 1
    shear_range = 0.2,  # Apply shear transformations
    zoom_range = 0.2    # Apply random zoom within range
)

# Data augmentation for validation data (only rescaling)
valid_datagen = ImageDataGenerator(
    rescale = 1.0/255  # Only rescale validation images, no augmentation
)

# -----------------------------------------
# Data Generators for Train and Validation
# -----------------------------------------

# Train generator using image_dataset_from_directory for training data
train_generator = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    image_size=(img_height, img_width),  # Resize images to 500x500 pixels
    batch_size=batch_size,
    label_mode='categorical'  # Multi-class classification
)

# Validation generator using image_dataset_from_directory for validation data
validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
    valid_directory,
    image_size=(img_height, img_width),  # Resize images to 500x500 pixels
    batch_size=batch_size,
    label_mode='categorical'  # Multi-class classification
)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# STEP 2.2 - MODEL BUILDING: Neural Network Architecture Design (30 Marks)
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# First Model: Simpler CNN Model
model_1 = Sequential([
    Conv2D(32, (3, 3), 
           activation='relu', 
           input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), 
           activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compile the First Model:
model_1.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

# Second Model: More Complex CNN Model
model_2 = Sequential([
    Conv2D(64, (3, 3), 
           activation='relu', 
           input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), 
           activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(256, (3, 3), 
           activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compile the Second Model:
model_2.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

# Print Summaries for Both Models:
print("Model 1 Summary:")
model_1.summary()

print("\nModel 2 Summary:")
model_2.summary()





























