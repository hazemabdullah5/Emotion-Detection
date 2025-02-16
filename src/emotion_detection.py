import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def train_emotion_model(
    train_dir=None,
    val_dir=None,
    test_dir=None,
    batch_size=16,
    num_classes=7,
    epochs=50,
    img_height=48,
    img_width=48,
    learning_rate=0.0005
):
    """
    Train a basic CNN model for emotion detection using images located in
    train, validation, and test directories.

    :param train_dir: Path to the training images folder.
    :param val_dir: Path to the validation images folder.
    :param test_dir: Path to the test images folder.
    :param batch_size: Batch size for training.
    :param num_classes: Number of emotion classes.
    :param epochs: Number of training epochs.
    :param img_height: Input image height.
    :param img_width: Input image width.
    :param learning_rate: Learning rate for the Adam optimizer.
    :return: (model, history) - trained Keras model and its training history.
    """

    # -------------------------------------------------------------
    # 1. Set default paths if none provided
    # -------------------------------------------------------------
    # Using relative paths to go one level up from 'src' folder,
    # assuming your data folder is at the same level as 'src'.
    current_dir = os.path.dirname(__file__)
    if train_dir is None:
        train_dir = os.path.join(current_dir, "../data/train")
    if val_dir is None:
        val_dir = os.path.join(current_dir, "../data/validation")
    if test_dir is None:
        test_dir = os.path.join(current_dir, "../data/test")

    # -------------------------------------------------------------
    # 2. Data Generators
    # -------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(img_height, img_width),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        directory=val_dir,
        target_size=(img_height, img_width),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(img_height, img_width),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # -------------------------------------------------------------
    # 3. Build the CNN Model
    # -------------------------------------------------------------
    model = Sequential()

    # Convolutional block 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Convolutional block 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Convolutional block 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Convolutional block 4
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Flatten
    model.add(Flatten())

    # Dense layers
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation='softmax'))

    # -------------------------------------------------------------
    # 4. Compile the Model
    # -------------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # -------------------------------------------------------------
    # 5. Train the Model
    # -------------------------------------------------------------
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )

    # -------------------------------------------------------------
    # 6. Evaluate the Model
    # -------------------------------------------------------------
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print("Test Loss: {:.4f}".format(test_loss))
    print("Test Accuracy: {:.4f}".format(test_acc))

    # -------------------------------------------------------------
    # 7. Save the Model
    # -------------------------------------------------------------
    models_dir = os.path.join(current_dir, "../models")
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, "emotion_model.h5"))

    # -------------------------------------------------------------
    # 8. Plot Training Curves (Optional)
    # -------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    plt.show()

    return model, history

if __name__ == "__main__":
    # If you run this file directly (e.g., `python emotion_detection.py`),
    # it will train the model with default directories and hyperparameters.

    train_emotion_model()
