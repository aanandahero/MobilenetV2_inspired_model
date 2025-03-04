import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataset_dir = r"C:\Users\Aananda Sagar Thapa\OneDrive\Desktop\ASL_Alphabet_Dataset\asl_alphabet_train"

# Image properties
IMG_SIZE = (128, 128)
BATCH_SIZE = 32


datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# Load the Model

try:
    model = load_model("asl_model_high_mobilenet5555.h5")
    print("Model loaded successfully!")
except OSError:
    print("Model file not found! Make sure the model is saved correctly.")
    exit()


# Evaluate Model on Validation Data

test_loss, test_accuracy = model.evaluate(val_generator, verbose=1)

# Print results
print(f"Validation Loss: {test_loss:.4f}")
print(f"Validation Accuracy: {test_accuracy:.4f}")


# Generate Predictions

predictions = model.predict(val_generator)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())


print("\nSample Predictions:")
for i in range(min(10, len(actual_classes))):
    print(f"Actual: {class_labels[actual_classes[i]]}, Predicted: {class_labels[predicted_classes[i]]}")

# ==============================
# Generate Confusion Matrix
# ==============================
cm = confusion_matrix(actual_classes, predicted_classes)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print("\n Classification Report:")
print(classification_report(actual_classes, predicted_classes, target_names=class_labels))

# ==============================
# Plot Accuracy & Loss Graphs
# ==============================
try:
    with open("training_history_mobilenet5555.pkl", "rb") as f:
        history = pickle.load(f)

    # Check if history keys exist
    if "accuracy" in history and "val_accuracy" in history:
        # Plot Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(history["accuracy"], label="Training Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Over Epochs")
        plt.legend()
        plt.show()
    else:
        print(" No 'accuracy' data found in training history.")

    if "loss" in history and "val_loss" in history:
        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(history["loss"], label="Training Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Model Loss Over Epochs")
        plt.legend()
        plt.show()
    else:
        print("No 'loss' data found in training history.")

except FileNotFoundError:
    print(" No saved training history found. Make sure to save history when training.")
