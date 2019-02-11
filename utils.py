import numpy as np
import pandas as pd
import os
import cv2
from model import *
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def load_dataset(path, height, width):
    image_list, label_list = [], []
    for root, dirs, files in os.walk(path):
        for file in files:
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (height, width))
            image_list.append(image)
            label_list.append(int(root[-3:])-1)
            
    num_classes = len(np.unique(label_list))
    
    return image_list, label_list, num_classes

def load_textdata(path):
    cols = ['pokedex_number', 'name', 'generation']
    df = pd.read_csv(path, usecols=cols)
    df = df[df['generation']==1].drop(labels=['generation'], axis=1)
    
    return df
    
def train_model(model, epochs, save_path, X_train, y_train, X_dev, y_dev, data_augmentation=True):
    
    batch_size = 32
    early_stopping = EarlyStopping(patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint(save_path, save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
    callbacks = [early_stopping, model_checkpoint, reduce_lr]
    
    if data_augmentation:
        datagen = ImageDataGenerator(
            rescale=1/255.,
            featurewise_center=False, # Set input mean to 0 over the dataset, feature-wise.
            samplewise_center=False, # Set each sample mean to 0.
            featurewise_std_normalization=False, # Divide inputs by std of the dataset, feature-wise.
            samplewise_std_normalization=False, # Divide each input by its std.
            zca_whitening=False, # Apply ZCA whitening.
            rotation_range=25, # Degree range for random rotations.
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True)
#         data_gen.fit(X_train
    else:
        datagen = ImageDataGenerator(
            rescale=1/255.)
        
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), 
                                  steps_per_epoch=len(X_train) / batch_size,
                                  validation_data=datagen.flow(X_dev, y_dev, batch_size=batch_size),
                                  validation_steps=len(X_dev) / batch_size,
                                  epochs=epochs,
                                  callbacks=callbacks)
    
    return model, history