import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Check TensorFlow and Keras Versions
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}\n")

# ---------------------------------------
# STEP 2.1 - DATA PROCESSING
# ---------------------------------------

# Define image dimensions and batch size
img_height = 500
img_width = 500
batch_size = 32

# Define Directory Paths for Training, Validation, and Test Data
train_directory = 'DataSet/train'  # Relative path to train folder
valid_directory = 'DataSet/valid'  # Relative path to validation folder
test_directory = 'DataSet/test'    # Relative path to test folder

# Data Augmentation for Training Data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values between 0 and 1
    shear_range=0.2,  # Apply shear transformations
    zoom_range=0.2    # Apply random zoom within range
)

# Data augmentation for validation data (only rescaling)
valid_datagen = ImageDataGenerator(
    rescale=1./255  # Only rescale validation images, no augmentation
)

# Data Generators for Train and Validation
train_generator = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
    valid_directory,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

# Get number of steps per epoch and validation steps as integers
steps_per_epoch = int(train_generator.cardinality() // batch_size)  # Explicit integer conversion
validation_steps = int(validation_generator.cardinality() // batch_size)  # Explicit integer conversion

print("\nTrain generator:", train_generator.cardinality().numpy(), "samples")
print("Validation generator:", validation_generator.cardinality().numpy(), "samples")
print("Class indices:", train_generator.class_names)

# ---------------------------------------
# STEP 2.2 - MODEL BUILDING
# ---------------------------------------

# Model 1: Simpler CNN Model
model_1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Regularization to reduce overfitting
    Dense(3, activation='softmax')  # 3 output classes
])

# Compile the Model
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model 2: More Complex CNN Model (same architecture as Model 1)
model_2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 output classes
])

# Compile the Model
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print Summaries for Both Models
print("\nModel 1 Summary:")
model_1.summary()
print("\nModel 2 Summary:")
model_2.summary()

# ---------------------------------------
# STEP 2.3 - MODEL TRAINING AND EVALUATION
# ---------------------------------------

# EarlyStopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Train the model (using model_1 as an example)
history = model_1.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,  # Correct value as integer
    epochs=15,  # Increase epochs, but use early stopping
    validation_data=validation_generator,
    validation_steps=validation_steps,  # Correct value as integer
    callbacks=[early_stopping]
)

# ---------------------------------------
# STEP 2.4 - MODEL EVALUATION
# ---------------------------------------

# Evaluate model performance on the test data
test_generator = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

test_loss, test_acc = model_1.evaluate(test_generator)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# ---------------------------------------
# STEP 2.5 - PLOTTING TRAINING & VALIDATION RESULTS
# ---------------------------------------

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




























'''
# Define the EarlyStopping callback for model training
def create_early_stopping_callback(
        monitor='val_loss',
        patience=4):
    
    return EarlyStopping(monitor=monitor, 
                         patience=patience, 
                         restore_best_weights=True)

early_stopping_callback = create_early_stopping_callback()

# Training the model
def train_model(model, train_generator, validation_generator, batch_size, epochs=10):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[early_stopping_callback]
    )
    return history

# Assuming the model is already compiled and ready to be trained
history = train_model(model, train_generator, validation_generator, batch_size=32)

# Step 3: Plotting the results: Training vs Validation Accuracy and Loss
def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Step 4: Model Evaluation
def evaluate_model(model, validation_generator):
    test_loss, test_acc = model.evaluate(validation_generator)
    print(f"Model's Final Accuracy on Validation Data: {test_acc:.4f}")

evaluate_model(model, validation_generator)
'''