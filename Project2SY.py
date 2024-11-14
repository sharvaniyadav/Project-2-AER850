
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Data Processing
# ------------------------------

# Define image dimensions and batch size
img_height = 500
img_width = 500
batch_size = 32


# Define Directory Paths for Training and Validation Data

train_directory = 'data/train'  # Relative path to train folder
valid_directory = 'data/valid'  # Relative path to validation folder
test_directory = 'data/test'    # Relative path to test folder
