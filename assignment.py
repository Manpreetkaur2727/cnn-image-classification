import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define dataset paths
train_dir = "/Users/manpreetkaur/Downloads/archive/seg_train/seg_train"
test_dir = "/Users/manpreetkaur/Downloads/archive/seg_test/seg_test"

# Data Augmentation and Normalization
datagen = ImageDataGenerator(
    rescale=1.0/255, 
    validation_split=0.2,  # 80-20 split
)

# Load Training & Validation Data
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",  # Ensures labels are one-hot encoded
    subset="training"
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Check Class Labels
print("Class indices:", train_generator.class_indices)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(6, activation="softmax")  # ✅ 6 output classes
])

# Compile Model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train Model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Evaluate Model on Test Set
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Classification Report
print("\nClassification Report:\n", classification_report(y_true, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=train_generator.class_indices.keys(), yticklabels=train_generator.class_indices.keys())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Plot Accuracy and Loss Graphs
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy")

# Loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Loss")

plt.show()

# Display Misclassified Images
misclassified_idx = np.where(y_pred_classes != y_true)[0]
plt.figure(figsize=(12,8))
for i, idx in enumerate(misclassified_idx[:9]):  # Show first 9 misclassified images
    img_path = test_generator.filepaths[idx]
    img = plt.imread(img_path)
    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(f"Predicted: {list(train_generator.class_indices.keys())[y_pred_classes[idx]]}\nActual: {list(train_generator.class_indices.keys())[y_true[idx]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
