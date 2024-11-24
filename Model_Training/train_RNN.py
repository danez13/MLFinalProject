import pandas as pd
import numpy as np
import math
import os
os.environ["KERAS_BACKEND"] = 'tensorflow'
import keras._tf_keras.keras as keras
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.layers import Dense,SimpleRNN,Dropout
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
from keras._tf_keras.keras.regularizers import l2

train_df = pd.read_csv(f"Processed_Data/train_FD001.csv")
test_df = pd.read_csv(f"Processed_Data/test_FD001.csv")

val_df = pd.read_csv(f"Processed_Data/train_FD003.csv")

sensors = ["s_1", "s_2", "s_3", "s_4", "s_5", "s_6", "s_7", "s_8", "s_9", "s_10", "s_11", "s_12", "s_13", "s_14", "s_15", "s_16","s_17","s_18","s_19","s_20","s_21"]

X_val = val_df[sensors].values
y_val = val_df[["failure_within_w1"]].values
X_val = np.expand_dims(X_val, axis=1)

# train with sensors
X_train = train_df[sensors].values
y_train = train_df[["failure_within_w1"]].values
X_train = np.expand_dims(X_train, axis=1)

# test with sensors
X_test = test_df[sensors].values
y_test = test_df[["failure_within_w1"]].values
X_test = np.expand_dims(X_test, axis=1)

model = Sequential([
    SimpleRNN(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(None,21)),
    Dropout(0.4),
    Dense(1, activation='sigmoid'),
])

# Specify training choices (optimizer, loss function, metrics)
model.compile(
    loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, batch_size=128, epochs=1024, validation_data=(X_val,y_val))

results = model.evaluate(X_test,y_test)
print(f"loss: {results[0]}")
print(f"accuracy: {results[1]}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


