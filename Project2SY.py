import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# Check TensorFlow and Keras Versions (Shashank did it in sample code):
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

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

train_directory = 'DataSet/train'  # Relative path to train folder
valid_directory = 'DataSet/valid'  # Relative path to validation folder
test_directory = 'DataSet/test'    # Relative path to test folder

# IMPORTANT NOTE TO SELF:
# Test and Validation and different.
# Test is used to evaluate the model's performance following model training and hyperparameter tuning!
# Validation is used to evaluate the model's perforoamnce after each epoch (training iteration) - utilized during training process!

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

# Train Generator Using image_dataset_from_directory for Training Data
train_generator = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    image_size=(img_height, img_width),  # Resize images to 500x500 pixels
    batch_size=batch_size,
    label_mode='categorical'  # Multi-class classification
)

# Validation Generator Using image_dataset_from_directory for Validation Data
validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
    valid_directory,
    image_size=(img_height, img_width),  # Resize images to 500x500 pixels
    batch_size=batch_size,
    label_mode='categorical'  # Multi-class classification
)

# Applied augmentation (shear, zoom) only to training data for better model generalization.
# Rescaled images for both training and validation data to normalize inputs.
# No augmentation on validation data to keep it unaltered and representative of real-world data.

# Print summary of data generator (number of samples in the train and validation sets)
print("Train generator:", train_generator.cardinality().numpy(), "samples")
print("Validation generator:", validation_generator.cardinality().numpy(), "samples")

# Check class labels to ensure they are split correctly
print("Class indices:", train_generator.class_names)

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
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
   
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Regularization to reduce overfitting
    Dense(3, activation='softmax')  # 3 output classes
])

# Compile the First Model:
model_1.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])


# Second Model: More Complex CNN Model

model_2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', 
    input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Regularization to reduce overfitting
    Dense(3, activation='softmax')  # 3 output classes
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






























