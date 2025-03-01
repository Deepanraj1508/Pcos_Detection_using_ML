import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2

# -------------------------------
# 🔹 1. Load and Preprocess Images
# -------------------------------
def load_and_preprocess_images(data_dir, img_size=64):
    images, labels = [], []
    class_names = os.listdir(data_dir)
    class_map = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = load_img(img_path, target_size=(img_size, img_size), color_mode='grayscale')
            img_array = img_to_array(img)

            # Convert to binary (thresholding)
            img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)[1]
            img_array = img_array / 255.0  # Normalize

            images.append(img_array)
            labels.append(class_map[class_name])

    return np.array(images), np.array(labels), class_names

# Load dataset
data_dir = r"image_model_train\dataset\train"
img_size = 64
X, y, class_names = load_and_preprocess_images(data_dir, img_size)

# Reshape data for ViT (grayscale images)
X = X.reshape(-1, img_size, img_size, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------
# 🔹 2. Define Vision Transformer (ViT)
# -----------------------------------
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformer_block(x, num_heads, embedding_dim, mlp_dim, dropout_rate):
    skip = x  # Skip connection
    
    # Layer Normalization
    x = layers.LayerNormalization()(x)

    # Multi-Head Self Attention
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim // num_heads)(x, x)

    # Ensure output shape matches input for skip connection
    x = layers.Dense(embedding_dim)(x)
    x = layers.Add()([skip, x])  # ✅ Fix shape mismatch

    # MLP Block
    skip = x  # Save skip connection
    x = layers.LayerNormalization()(x)
    x = mlp(x, [mlp_dim], dropout_rate)

    # Ensure output shape matches input for skip connection
    x = layers.Dense(embedding_dim)(x)
    x = layers.Add()([skip, x])  # ✅ Fix shape mismatch

    return x

def create_vit_model(image_size=64, patch_size=8, num_heads=8, transformer_layers=4, embedding_dim=256, mlp_dim=512, num_classes=10):
    inputs = keras.Input(shape=(image_size, image_size, 1))  # Grayscale input

    # Convert image into patches
    num_patches = (image_size // patch_size) ** 2
    patch_embedding = layers.Conv2D(filters=embedding_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    
    # Reshape patches
    patch_embedding = layers.Reshape((num_patches, embedding_dim))(patch_embedding)

    # ✅ Fix Position Embeddings Shape
    pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)(tf.range(num_patches))
    pos_embedding = tf.expand_dims(pos_embedding, axis=0)  # (1, num_patches, embedding_dim)

    # ✅ Ensure shapes match before addition
    x = layers.Add()([patch_embedding, pos_embedding])  

    # Transformer blocks
    for _ in range(transformer_layers):
        x = transformer_block(x, num_heads, embedding_dim, mlp_dim, dropout_rate=0.1)

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layers
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create model
    model = keras.Model(inputs, outputs)
    return model

# Create ViT model
vit_model = create_vit_model(image_size=img_size, num_classes=len(class_names))

# Compile Model
vit_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# -----------------------------------
# 🔹 3. Train Model
# -----------------------------------
history = vit_model.fit(
    X_train, y_train,
    batch_size=32,  # ✅ Explicit batch size
    epochs=20,
    validation_data=(X_test, y_test)
)

# Save Model
vit_model.save("image_binary_transformer_model.h5")

# Print Accuracy
train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")

print("✅ Model training complete and saved as image_binary_transformer_model.h5")
