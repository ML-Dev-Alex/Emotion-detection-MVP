from xml.parsers.expat import model
import keras
import os
import datetime
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from fer import FER
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix

train_dir = "database/train"
test_dir = "database/test"
# Original pixel size of the image (48x48x1), only 1 channel since they're grayscale
img_size = 48
classes = 7  # Number of emotions the model is able to classify


def count_images_per_class(path, set):
    '''
    Counts the number of images in each class folders and returns a dataframe with the results.

    path: The path to the base folder containing the train and test sets.
    set: train or test, selects which dataset to analyse.
    '''
    class_dict = {}
    for emotion in os.listdir(path):
        class_dir = f"{path}/{emotion}"
        class_dict[emotion] = len(os.listdir(class_dir))
    df = pd.DataFrame(class_dict, index=[set])
    return df


def get_model(input_size, classes=7):
    # Initialising the CNN
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
              activation='relu', input_shape=input_size))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
              padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    # Compliling the model
    model.compile(optimizer=Adam(learning_rate=0.0001, decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train(model_path='ferNet', best_weights='fernet_bestweight', epochs=60, batch_size=64):
    """
    Trains a deep learning model to identify emotions on pictures of faces.
    :param model_path: The path where the model will me saved.
    :param best_weights: The path where the weights will be saved.
    :param epochs: How long should the model train for.
    :param batch_size: How many images should the model evaluate per training step. Change this if running out of memory during training.
    :return: Nothing. 
    """

    if model_path is not None and model_path[-3:] != '.h5':
        model_path += ".h5"
    if best_weights is not None and best_weights[-3:] != '.h5':
        best_weights += ".h5"
    chk_path = model_path
    log_dir = "checkpoint/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    epochs = epochs
    batch_size = batch_size

    """
    Data Augmentation
    --------------------------
    rotation_range = rotates the image with the amount of degrees we provide
    width_shift_range = shifts the image randomly to the right or left along the width of the image
    height_shift range = shifts image randomly to up or below along the height of the image
    horizontal_flip = flips the image horizontally
    rescale = to scale down the pizel values in our image between 0 and 1
    zoom_range = applies random zoom to our object
    validation_split = reserves some images to be used for validation purpose
    """

    train_datagen = ImageDataGenerator(rotation_range=5,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       horizontal_flip=True,
                                       rescale=1./255,
                                       zoom_range=0.1,
                                       validation_split=0.2
                                       )
    validation_datagen = ImageDataGenerator(rescale=1./255,
                                            validation_split=0.2)

    """
    Applying data augmentation to the images as we read 
    them from their respectivve directories
    """
    train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                        target_size=(
                                                            img_size, img_size),
                                                        batch_size=64,
                                                        shuffle=True,
                                                        color_mode="grayscale",
                                                        class_mode="categorical",
                                                        subset="training"
                                                        )
    validation_generator = validation_datagen.flow_from_directory(directory=test_dir,
                                                                  target_size=(
                                                                      img_size, img_size),
                                                                  batch_size=64,
                                                                  shuffle=True,
                                                                  color_mode="grayscale",
                                                                  class_mode="categorical",
                                                                  subset="validation"
                                                                  )

    fernet = get_model((img_size, img_size, 1), classes)

    # Now we create some callback functions to improve the model training

    checkpoint = ModelCheckpoint(filepath=chk_path,
                                 save_best_only=True,
                                 verbose=1,
                                 mode='min',
                                 moniter='val_loss')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=6,
                                  verbose=1,
                                  min_delta=0.0001)

    csv_logger = CSVLogger('training.log')

    callbacks = [checkpoint, reduce_lr, csv_logger]

    fernet.compile(
        optimizer=Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # finally we train the model and save the weights

    steps_per_epoch = train_generator.n // train_generator.batch_size
    validation_steps = validation_generator.n // validation_generator.batch_size

    fernet.fit(x=train_generator,
               validation_data=validation_generator,
               epochs=epochs,
               batch_size=batch_size,
               callbacks=callbacks,
               steps_per_epoch=steps_per_epoch,
               validation_steps=validation_steps)

    fernet.save_weights(best_weights)


if __name__ == "__main__":
    # If you run this file, it will simply call the function with the default parameters.
    # You can also import it as a function and call the train function yourself.
    train()
