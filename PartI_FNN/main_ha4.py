import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras.datasets import mnist
from util import func_confusion_matrix
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, BatchNormalization, Activation


# load (downloaded if needed) the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# transform each image from 28 by 28 to a 784 pixel vector
pixel_count = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')
x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')

# normalize inputs from gray scale of 0-255 to values between 0-1
x_train = x_train / 255
x_test = x_test / 255

# Please write your own codes in responses to the homework assignment 4

# Split training data into training and validation sets (Question 1)
x_train_final, x_val, y_train_final, y_val = train_test_split( x_train, y_train, test_size = 10000, random_state = 42 )

# Define three different model architectures
def create_model_1():
    model = Sequential([
        Dense(512, activation = 'relu', input_shape = (784,)),
        Dropout(0.2),
        Dense(256, activation = 'relu'),
        Dense(10, activation = 'softmax')
    ])
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
                 loss = 'sparse_categorical_crossentropy',
                 metrics = ['accuracy'])
    return model

def create_model_2():
    model = Sequential([
        Dense(128, activation = 'tanh', input_shape = (784,)),
        Dense(64, activation = 'tanh'),
        Dense(10, activation = 'softmax')
    ])
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01),
                 loss = 'sparse_categorical_crossentropy',
                 metrics = ['accuracy'])
    return model

def create_model_3():
    model = Sequential([
        Dense(1024, activation = 'relu', input_shape = (784,)),
        Dropout(0.3),
        Dense(512, activation = 'relu'),
        Dropout(0.3),
        Dense(256, activation = 'relu'),
        Dense(10, activation = 'softmax')
    ])
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0005),
                 loss = 'sparse_categorical_crossentropy',
                 metrics = ['accuracy'])
    return model

def create_model_4():
    model = Sequential([
        Dense(512, activation='relu', input_shape=(784,)),
        Dropout(0.3),
        Dense(384, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def create_model_5():
    model = Sequential([
        Dense(512, activation='elu', input_shape=(784,)),
        Dropout(0.2),
        Dense(256, activation='elu'),
        Dense(128, activation='elu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def create_model_6():
    model = Sequential([
        Dense(512, input_shape=(784,)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Function to evaluate model and print metrics
def evaluate_model(model, x_data, y_true, set_name=""):
    y_pred = model.predict(x_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    conf_matrix, accuracy, recall, precision = func_confusion_matrix(y_true, y_pred_classes)
    
    print(f"\nResults for {set_name}:")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Average Accuracy: {accuracy:.4f}")
    print("\nPer-class Precision:")
    for i, p in enumerate(precision): print(f"Class {i}: {p:.4f}")
    print("\nPer-class Recall:")
    for i, r in enumerate(recall): print(f"Class {i}: {r:.4f}")
    
    return accuracy, y_pred_classes

def plot_incorrect_predictions(x_test_data, y_test_true, y_test_pred, num_examples=10):
    incorrect_mask = y_test_true != y_test_pred
    incorrect_indices = np.where(incorrect_mask)[0]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for idx, ax in enumerate(axes):
        if idx < num_examples:
            image_idx = incorrect_indices[idx]
            image = x_test_data[image_idx].reshape(28, 28)
            ax.imshow(image, cmap='gray')
            ax.set_title(f'True: {y_test_true[image_idx]}\nPred: {y_test_pred[image_idx]}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Train and evaluate all models
models = [
    ("Model 1 (Original)", create_model_1()),
    ("Model 4 (Deeper)", create_model_4()),
    ("Model 5 (ELU)", create_model_5()),
    ("Model 6 (BatchNorm)", create_model_6())
]

# Dictionary to store validation results
val_results = {}

# Train and evaluate each model
for name, model in models:
    print(f"\nTraining {name}")
    history = model.fit(
        x_train_final, y_train_final,
        epochs=10,
        batch_size=128,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    val_accuracy, _ = evaluate_model(model, x_val, y_val, f"Validation Set - {name}")
    val_results[name] = (val_accuracy, model)

# Find the best model
best_model_name = max(val_results.keys(), key=lambda k: val_results[k][0])
best_model = val_results[best_model_name][1]
print(f"\nBest model: {best_model_name}")

# Evaluate best model on test set (Question 2)
test_accuracy, test_predictions = evaluate_model(best_model, x_test, y_test, "Test Set")

# Visualize incorrect predictions
plot_incorrect_predictions(x_test, y_test, test_predictions)