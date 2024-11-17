
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Check TensorFlow and Keras Versions
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}\n")

# -------------------------
# -------------------------
# STEP 1 - DATA PROCESSING
# -------------------------
# -------------------------

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
    horizontal_flip = True
)

# Data augmentation for validation data (only rescaling)
valid_datagen = ImageDataGenerator(
    rescale=1./255  # Only rescale validation images, no augmentation
)

# Data Generators for Train and Validation
train_generator = train_datagen.flow_from_directory (
    train_directory,
    image_size=(img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical'
)

validation_generator = valid_datagen.flow_from_directory (
    valid_directory,
    image_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode ='categorical'
)

print("\nTrain generator:", train_generator.cardinality().numpy(), "samples")
print("Validation generator:", validation_generator.cardinality().numpy(), "samples")
print("Class indices:", train_generator.class_names)

# --------------------------------------------
# --------------------------------------------
# STEP 2 - NEURAL NETWORK ARCHITECTURE DESIGN
# --------------------------------------------
# --------------------------------------------

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
dcnn_model.add(layers.Dropout(0.25))  # Dropout to reduce overfitting

# Layer 3: Convolution with 128 filters and MaxPooling
dcnn_model.add(layers.Conv2D(128, (3, 3), 
                             activation='relu'))
dcnn_model.add(layers.MaxPooling2D((2, 2)))
dcnn_model.add(layers.Dropout(0.3))  # Dropout to reduce overfitting

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
dcnn_model.add(layers.Dropout(0.5))  # Dropout for regularization

# ---------------------------------------
# ---------------------------------------
# STEP 3 - MODEL HYPERPARAMETER ANALYSIS
# ---------------------------------------
# ---------------------------------------

dcnn_model.compile(optimizer='adam', 
                   loss="categorical_crossentropy", 
                   metrics=["accuracy"])

dcnn_model.summary()










































'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Check TensorFlow and Keras Versions (Shashank did it in sample code):
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}\n")

# ---------------------------------------
# ---------------------------------------
# STEP 2.1 - DATA PROCESSINGh
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
    rescale = 1./255,  # Normalize pixel values between 0 and 1
    shear_range = 0.2,  # Apply shear transformations
    zoom_range = 0.2    # Apply random zoom within range
)

# Data augmentation for validation data (only rescaling)
valid_datagen = ImageDataGenerator(
    rescale = 1./255  # Only rescale validation images, no augmentation
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
print("\nTrain generator:", train_generator.cardinality().numpy(), "samples")
print("Validation generator:", validation_generator.cardinality().numpy(), "samples")

# Check class labels to ensure they are split correctly
print("Class indices:", train_generator.class_names)

# --------------------------------------------------------------
# --------------------------------------------------------------
# STEP 2.2 - MODEL BUILDING: Neural Network Architecture Design 
# --------------------------------------------------------------
# --------------------------------------------------------------

# First Model: Simpler CNN Model
model_1 = Sequential([
    Conv2D(32, (3, 3), 
           activation='relu', 
           input_shape = (img_height, img_width, 3)),
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
    input_shape = (img_height, img_width, 3)),
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
print("\nModel 1 Summary:")
model_1.summary()

print("\nModel 2 Summary:")
model_2.summary()

# ---------------------------------------
# STEP 2.3 - MODEL TRAINING AND EVALUATION
# ---------------------------------------

# EarlyStopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=4, 
                               restore_best_weights=True)


# Train the model (using model_2 as an example)
history = model_1.fit(
    train_generator,
    steps_per_epoch=train_generator.cardinality().numpy(),  # Updated to use cardinality()
    epochs = 15,  # Increase epochs, but use early stopping
    validation_data = validation_generator,
    validation_steps = validation_generator.cardinality().numpy(),  # Updated to use cardinality()
    callbacks=[early_stopping] 
)


# -----------------------------
# -----------------------------
# STEP 2.4 - MODEL EVALUATION
# -----------------------------
# -----------------------------

# Evaluate model performance on the test data
test_generator = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

test_loss, test_acc = model_2.evaluate(test_generator)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# --------------------------------------------------
# --------------------------------------------------
# STEP 2.5 - PLOTTING TRAINING & VALIDATION RESULTS
# --------------------------------------------------
# --------------------------------------------------

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

'''




























