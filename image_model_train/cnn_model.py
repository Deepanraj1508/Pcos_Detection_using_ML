from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt

# Define LSTM-based model
model_lstm = Sequential()

# Reshape input to 1D sequence (224x224 pixels, 3 channels flattened)
model_lstm.add(Reshape((224, 224*3), input_shape=(224, 224, 3)))

# Add LSTM layers
model_lstm.add(LSTM(128, return_sequences=True, activation='relu'))
model_lstm.add(LSTM(64, return_sequences=False, activation='relu'))

# Fully connected layers
model_lstm.add(Dense(64, activation='relu'))
model_lstm.add(Dense(2, activation='softmax'))  # Output layer (2 classes)

# Compile the model
model_lstm.compile(
    optimizer=Adam(),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
history_lstm = model_lstm.fit(
    train_it,  # Training dataset
    validation_data=val_it,  # Validation dataset
    epochs=8
)

# Evaluate the model on validation data
val_loss, val_accuracy = model_lstm.evaluate(val_it)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the trained model
model_lstm.save("lstm_model.h5")
print("Model saved as lstm_model.h5")

# Plot training vs validation loss
plt.plot(history_lstm.history['loss'], label='Train Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.show()
