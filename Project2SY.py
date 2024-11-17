# Importing Necessary Libraries:
    
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check TensorFlow and Keras Versions
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}\n")

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# STEP 2.1 - DATA PROCESSING
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
# STEP 2.2 - NEURAL NETWORK ARCHITECTURE DESIGN
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

# Dense Layers

dcnn_model.add(layers.Dense(64, 
                            activation='relu'))  # Fully connected layer
dcnn_model.add(layers.Dropout(0.5))  # Dropout for regularization

# Dense Layer for multi-class classification (3 classes)

dcnn_model.add(layers.Dense(3, activation='softmax'))  # Output layer with 3 classes (softmax)

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# STEP 2.3 - MODEL HYPERPARAMETER ANALYSIS
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

dcnn_model.compile(optimizer='adam', 
                   loss="categorical_crossentropy", 
                   metrics=["accuracy"])

# Display the model architecture summary to verify the layers, parameters, and output shapes
dcnn_model.summary()

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# STEP 2.4 - MODEL TRAINING
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

# Train the model using the data generators for training and validation
training_history = dcnn_model.fit(
    train_generator,  # Using the train data generator
    epochs=60,        # Number of epochs for training
    validation_data = validation_generator)  # Using the validation data generator

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# STEP 2.5 - MODEL EVALUATION
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

# Evaluate the model performance on the validation data
val_loss, val_accuracy = dcnn_model.evaluate(validation_generator)
print(f"\nValidation accuracy: {val_accuracy:.4f}")
print(f"Validation loss: {val_loss:.4f}")

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# STEP 2.6: Plotting Training & Validation Results
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

# Plot training & validation accuracy graph size
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.plot(training_history.history['accuracy'], label="Training Accuracy")
plt.plot(training_history.history['val_accuracy'], label="Validation Accuracy")

# Plot Aesthetics
plt.title('Model Training and Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot training & validation loss graph size
plt.figure(figsize=(12, 4))

# Plotting the training vs validation loss
plt.plot(training_history.history['loss'], label="Training Loss")
plt.plot(training_history.history['val_loss'], label="Validation Loss")

# Plot Aesthetics
plt.title('Model Training and Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()


# Save the model to later be utlized 2.7, otherwise known as step 5 from the project
dcnn_model.save("Aircraft_DCNN_Model.keras")

plt.show()





