# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np

from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import RandomForestClassifier

import keras
from keras.layers import Input, Dense, Dropout, Activation, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras import optimizers

from failing_loudly.shared_utils import *
import os

import keras_resnet
import keras_resnet.models

import pickle


# -------------------------------------------------
# SHIFT REDUCTOR
# -------------------------------------------------


class ShiftReductor:

    def __init__(self, X, y, X_val, y_val, dr_tech, orig_dims, datset, dr_amount=None, var_ret=0.8):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.dr_tech = dr_tech
        self.orig_dims = orig_dims
        self.datset = datset
        self.mod_path = None

        # We can set the number of dimensions automatically by computing PCA's variance retention rate.
        if dr_amount is None:
            pca = PCA(n_components=var_ret, svd_solver='full')
            pca.fit(X)
            self.dr_amount = pca.n_components_
        else:
            self.dr_amount = dr_amount

    # Since the autoencoder's and ResNet's training procedure can take some time, we usually only train them once
    # and save the model for subsequent uses of dimensionality reduction. If we can't find a corresponding model in
    # the usual directory, then we train the respective model on the fly. PCA and SRP are always trained on the fly.
    def fit_reductor(self):
        if not os.path.exists('./saved_models/'):
            os.makedirs('./saved_models/')
        if self.dr_tech == DimensionalityReduction.PCA:
            return self.principal_components_anaylsis()
        elif self.dr_tech == DimensionalityReduction.SRP:
            return self.sparse_random_projection()
        elif self.dr_tech == DimensionalityReduction.UAE:
            self.mod_path = './saved_models/' + self.datset + '_untr_autoencoder_model.h5'
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path)
            return self.autoencoder(train=False)
        elif self.dr_tech == DimensionalityReduction.TAE:
            self.mod_path = './saved_models/' + self.datset + '_autoencoder_model.h5'
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path)
            return self.autoencoder(train=True)
        elif self.dr_tech == DimensionalityReduction.BBSDs:
            self.mod_path = './saved_models/' + self.datset + '_standard_class_model.h5'
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path, custom_objects=keras_resnet.custom_objects)
            return self.neural_network_classifier(train=True)
        elif self.dr_tech == DimensionalityReduction.BBSDh:
            self.mod_path = './saved_models/' + self.datset + '_standard_class_model.h5'
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path, custom_objects=keras_resnet.custom_objects)
            return self.neural_network_classifier(train=True)
        elif self.dr_tech == DimensionalityReduction.BBSDs_RF:
            self.mod_path = './saved_models/' + self.datset + '_rf_class_model.h5'
            if os.path.exists(self.mod_path):
                return pickle.load(open(self.mod_path, 'rb'))
            return self.random_forest_classifier()
        elif self.dr_tech == DimensionalityReduction.BBSDh_RF:
            self.mod_path = './saved_models/' + self.datset + '_rf_class_model.h5'
            if os.path.exists(self.mod_path):
                return pickle.load(open(self.mod_path, 'rb'))
            return self.random_forest_classifier()

    # Given a model to reduce dimensionality and some data, we have to perform different operations depending on
    # the DR method used.
    def reduce(self, model, X):
        if self.dr_tech == DimensionalityReduction.PCA or self.dr_tech == DimensionalityReduction.SRP:
            return model.transform(X)
        elif self.dr_tech == DimensionalityReduction.UAE or self.dr_tech == DimensionalityReduction.TAE or \
                self.dr_tech == DimensionalityReduction.BBSDs:
            if isinstance(self.orig_dims, tuple):
                X = X.reshape(len(X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2])
            pred = model.predict(X)
            pred = pred.reshape((len(pred), np.prod(pred.shape[1:])))
            return pred
        elif self.dr_tech == DimensionalityReduction.NoRed:
            return X
        elif self.dr_tech == DimensionalityReduction.BBSDh:
            if isinstance(self.orig_dims, tuple):
                X = X.reshape(len(X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2])
            pred = model.predict(X)
            pred = np.argmax(pred, axis=1)
            return pred
        elif self.dr_tech == DimensionalityReduction.BBSDs_RF:
            if isinstance(self.orig_dims, tuple):
                X = X.reshape(len(X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2])
            pred = model.predict_proba(X)
            return pred
        elif self.dr_tech == DimensionalityReduction.BBSDh_RF:
            if isinstance(self.orig_dims, tuple):
                X = X.reshape(len(X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2])
            pred = model.predict(X)
            pred = np.expand_dims(pred, axis=1)
            return pred

    def sparse_random_projection(self):
        srp = SparseRandomProjection(n_components=self.dr_amount)
        srp.fit(self.X)
        return srp

    def principal_components_anaylsis(self):
        pca = PCA(n_components=self.dr_amount)
        pca.fit(self.X)
        return pca

    # We construct a couple of different autoencoder architectures depending on the individual dataset. This is due to
    # different input shapes. We usually share architectures if the shapes of two datasets match.
    def autoencoder(self, train=False):
        X = self.X.reshape(len(self.X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2])

        input_img = Input(shape=self.orig_dims)

        # Define various architectures.
        if self.datset == 'mnist' or self.datset == 'fashion_mnist':
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)

            x = Conv2D(2, (3, 3), activation='relu', padding='same')(encoded)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(16, (3, 3), activation='relu')(x)
            x = UpSampling2D((2, 2))(x)
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        elif self.datset == 'cifar10' or self.datset == 'cifar10_1' or self.datset == 'coil100' or self.datset == 'svhn':

            x = Conv2D(64, (3, 3), padding='same')(input_img)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(16, (3, 3), padding='same')(x)
            x = Activation('relu')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)

            x = Conv2D(16, (3, 3), padding='same')(encoded)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(32, (3, 3), padding='same')(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(64, (3, 3), padding='same')(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(3, (3, 3), padding='same')(x)
            decoded = Activation('sigmoid')(x)

        elif self.datset == 'mnist_usps':
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
            encoded = MaxPooling2D((2, 2), padding='same')(x)

            x = Conv2D(2, (3, 3), activation='relu', padding='same')(encoded)
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        # Construct both an encoding model and a full encoding-decoding model. The first one will be used for mere
        # dimensionality reduction, while the second one is needed for training.
        encoder = Model(input_img, encoded)
        autoenc = Model(input_img, decoded)

        autoenc.compile(optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9), loss='binary_crossentropy')

        if train:
            autoenc.fit(self.X.reshape(len(self.X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]),
                        self.X.reshape(len(self.X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]),
                        epochs=200,
                        batch_size=128,
                        validation_data=(
                        self.X_val.reshape(len(self.X_val), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]),
                        self.X_val.reshape(len(self.X_val), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2])),
                        shuffle=True)

        encoder.save(self.mod_path)

        return encoder

    # Our label classifier constitutes of a simple ResNet-18.
    def neural_network_classifier(self, train=True):
        if len(self.X.shape) == 2:  # tabular data
            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
            early_stopper = EarlyStopping(min_delta=0.001, patience=10)
            batch_size = 128
            nb_classes = len(np.unique(self.y))
            epochs = 100
            y_loc = np_utils.to_categorical(self.y, nb_classes)
            y_val_loc = np_utils.to_categorical(self.y_val, nb_classes)

            model = Sequential()
            model.add(Dense(64, input_dim=self.orig_dims, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(nb_classes, activation='softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])

            model.fit(self.X, y_loc,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(self.X_val, y_val_loc),
                      shuffle=True,
                      callbacks=[lr_reducer, early_stopper])
        else:
            D = self.X.shape[1]

            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
            early_stopper = EarlyStopping(min_delta=0.001, patience=10)
            batch_size = 128
            nb_classes = len(np.unique(self.y))
            epochs = 200
            y_loc = np_utils.to_categorical(self.y, nb_classes)
            y_val_loc = np_utils.to_categorical(self.y_val, nb_classes)

            model = keras_resnet.models.ResNet18(keras.layers.Input(self.orig_dims), classes=nb_classes)
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9),
                          metrics=['accuracy'])

            model.fit(self.X.reshape(len(self.X), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]), y_loc,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(
                      self.X_val.reshape(len(self.X_val), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]),
                      y_val_loc),
                      shuffle=True,
                      callbacks=[lr_reducer, early_stopper])

        model.save(self.mod_path)

        return model

    def random_forest_classifier(self):

        if len(self.X.shape) == 2:  # tabular data

            nb_classes = len(np.unique(self.y))

            n_estimators = 100
            criterion = 'gini'
            max_depth = None

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)

            model.fit(self.X, self.y)

        else:
            raise NotImplementedError

        pickle.dump(model, open(self.mod_path, 'wb'))

        return model
