# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# STEP 2.7 - MODEL TESTING
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

# Import necessary libraries
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

class ModelTester:
    def __init__(self, model_path, img_width=500, img_height=500):
        
        # Initialize the model tester with the model path and image dimensions
        self.model = load_model(model_path)  # Load the pre-trained model
        self.img_width = img_width
        self.img_height = img_height
        self.class_labels = ['Crack', 'Missing Head', 'Paint-Off']  # Define the class labels

    def preprocess_image(self, image_path):
        
        # Preprocess the image by resizing and normalizing
        img = load_img(image_path, target_size=(self.img_width, self.img_height))  # Load image
        img_array = img_to_array(img)  # Convert image to array
        img_array /= 255  # Rescale image pixel values to range [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    def make_prediction(self, img_array):
        
        # Make a prediction using the trained model
        return self.model.predict(img_array)  # Return prediction

    def plot_results(self, image_path, predictions, true_label):
        # Display the image and the prediction results
        predicted_label = self.class_labels[np.argmax(predictions)]  # Get the predicted label

        fig, ax = plt.subplots(figsize=(6, 6))  # Set figure size for display
        img = plt.imread(image_path)  # Load image again for display
        plt.imshow(img)  # Display image
        ax.axis('off')  # Hide axis

        # Dynamically create the title based on true and predicted labels
        title = f"True {true_label} Classification Label: {true_label}\n" \
                f"Predicted {predicted_label} Classification Label: {predicted_label}"
        plt.title(title)  # Set title for the plot

        # Sort predictions and display the probability of each class
        sorted_labels = sorted(zip(self.class_labels, predictions[0]), key=lambda x: x[1], reverse=True)
        for index, (label, prob) in enumerate(sorted_labels):
            ax.text(10, 25 + index * 30, f"{label}: {prob * 100:.2f}%", bbox=dict(facecolor="pink"),
                    fontsize=10, color='black')  # Add text with probability

        # Add your name and student number at the bottom of the image
        plt.figtext(0.5, 0.01, "Sharvani Yadav - 501108658", ha="center", fontsize=12, color="black", 
                    bbox=dict(facecolor="white", edgecolor="none"))

        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()  # Display the plot

    def test_image(self, image_path, true_label):
        # Process a single image and display the prediction results
        img_array = self.preprocess_image(image_path)  # Preprocess the image
        predictions = self.make_prediction(img_array)  # Make predictions
        self.plot_results(image_path, predictions, true_label)  # Display the results

# -----------------------------------------------------------------------------------------

# Specify the test images (replace with actual paths as needed)
test_images = [
    (r"C:\Users\Sharvani Yadav\OneDrive\Documents\GitHub\Project-2-AER850\DataSet\test\crack\test_crack.jpg", "Crack"),
    (r"C:\Users\Sharvani Yadav\OneDrive\Documents\GitHub\Project-2-AER850\DataSet\test\missing-head\test_missinghead.jpg", "Missing Head"),
    (r"C:\Users\Sharvani Yadav\OneDrive\Documents\GitHub\Project-2-AER850\DataSet\test\paint-off\test_paintoff.jpg", "Paint-Off")]

# -----------------------------------------------------------------------------------------

# Create an instance of the ModelTester class
tester = ModelTester("Aircraft_DCNN_Model.keras")

# Test each image in the test_images list
for image_path, true_label in test_images:
    tester.test_image(image_path, true_label)
