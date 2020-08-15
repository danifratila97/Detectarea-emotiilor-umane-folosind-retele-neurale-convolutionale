from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import os



os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 5 clase  - 'Angry' , 'Happy' , 'Neutral' , 'Sad' , 'Surprised'
nr_clase = 5
class_names = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
img_rows, img_cols = 48, 48
batch_size = 32

#path-urile catre directoarele ce contin imaginile de test si de validare
train_data_dir = '/home/dani/Licenta/datasets/fer2013/train'
validation_data_dir = '/home/dani/Licenta/datasets/fer2013/validation'


# data augmentation pentru o baza de date mai mare

def augmentare():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')
    return train_datagen


def normare():
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    return validation_datagen


def generateTrainSet():
    train_datagen = augmentare()
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode='grayscale',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    return train_generator


def generateValidationSet():
    validation_datagen = normare()
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode='grayscale',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    return validation_generator


def buildModel():
    model = Sequential()

    # Bloc convolutional (1)

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols, 1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(img_rows, img_cols, 1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Bloc convolutional (2)

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Bloc convolutional (3)

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Bloc convolutional (4)

    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Bloc flatten (5)

    model.add(Flatten())
    model.add(Dense(64, kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Bloc fully connected (6)

    model.add(Dense(64, kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Bloc fully connected clasificator (7)

    model.add(Dense(nr_clase, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))

    return model


from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def generate_callbacks(modelName, patience_es, patience_lr):
    checkpoint = ModelCheckpoint(modelName,
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True,
                                 verbose=1)

    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=patience_es,
                              verbose=1,
                              restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=patience_lr,
                                  verbose=1,
                                  min_delta=0.0001)

    callbacks = [earlystop, checkpoint, reduce_lr]

    return callbacks


def compileModel(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])


def antrenareModel(model, train_set, validation_set, nb_train_samples, nb_validation_samples, epochs, callbacks):
    history = model.fit_generator(
        train_set,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_set,
        validation_steps=nb_validation_samples // batch_size)

    return history


def plotAccuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


def plotLoss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


def printModelArchitecture(model, filename):
    plot_model(
        model,
        to_file=filename,
        show_shapes=True,
        show_layer_names=True,
        expand_nested=False,
        dpi=196,
    )


def main():
    nb_train_samples = 24256
    nb_validation_samples = 3006
    epochs = 25

    train_set = generateTrainSet()
    validation_set = generateValidationSet()
    callbacks = generate_callbacks('Emotion_train_model2.h5', 9, 3)

    print(emotion_model.summary())

    compileModel(emotion_model)

    history = antrenareModel(emotion_model,
                             train_set,
                             validation_set,
                             nb_train_samples,
                             nb_validation_samples,
                             epochs,
                             callbacks)

    plotAccuracy(history)
    plotLoss(history)
    printModelArchitecture(emotion_model, "model.png")

if __name__ == "__main__":
    main()
