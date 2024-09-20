from __future__ import absolute_import, division, print_function, unicode_literals
import os
import scipy
import scipy.io as sio
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import copy

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras import regularizers
from numpy.random import seed
seed(40)
tf.random.set_seed(41)


def plot_history(history, n):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
    plt.ylim([0, 0.0000006])
    plt.legend()
    # plt.show()
    plt.savefig('../data/plots/network' + str(n) + '.png', dpi=300)
    plt.close('all')

def plot_pred(test, predictions, n):
    plt.scatter(y_test[:, 5], test_predictions[:, 5])
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.savefig('../data/plots/pred_last_el_' + str(n) + '.png', dpi=300)
    plt.close('all')


def build_model(num_layers, num_neurons):
    inputs = keras.Input(shape=([normed_x_train.shape[1]]), name='input')

    for k in range(0, num_layers):
        if k == 0:
            hl = layers.Dense(num_neurons, activation='relu', name=str(k+1))(inputs)
        else:
            hl = layers.Dense(num_neurons, activation='relu', name=str(k+1))(hl)
    outputs = layers.Dense(6, name='output', activation='relu')(hl)

    nn = keras.Model(inputs=inputs, outputs=outputs, name='learn_B')

    optimizer = tf.keras.optimizers.RMSprop(0.00001)
    nn.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return nn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# path = "/Users/claudio/Desktop/PhD/Dataset/DataFromCluster/25_foreach_ne_6_10k/"
path = "/home/tomasi/tensorflow/learn_multigrid/data/"
x = sio.loadmat(path+'x_not_val.mat')
y = sio.loadmat(path+'y_not_val.mat')
x = x['x']
y = y['y']
val_x = sio.loadmat(path+'val_x.mat')
val_y = sio.loadmat(path+'val_y.mat')
val_x = val_x['val_x']
val_y = val_y['val_y']

normed_x_train, normed_x_test, y_train, y_test = train_test_split(x, y, random_state=42)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# The patience parameter is the amount of epochs to check for improvement
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7,
#                               patience=6, min_lr=0.00000001)


number = 0
column_names = ["#", "layers", "neurons", "TestMAE", "TestMSE", "TestLoss", "% Error"]
# df = pd.DataFrame(columns = column_names)
df = pd.DataFrame()
best_loss = 1
for i in range(2, 18):
    n_layers = i+1

    for j in np.arange(100, 801, 50):
        number += 1
        n_neurons = j

        model = build_model(n_layers, n_neurons)
        model.summary()

        history = model.fit(normed_x_train, y_train, epochs=600, batch_size=32,
                            validation_data=(val_x, val_y), verbose=1, callbacks=[early_stop])
        #
        loss, mae, mse = model.evaluate(normed_x_test, y_test, verbose=0)
        #
        # print("Testing set Mean Abs Error:", mae)
        # print("Testing set Mean Squared Error:", mse)
        # print("Testing set Loss:", loss)
        error = mae/np.mean(y_test)
        # print("% error:", error)
        #
        plot_history(history, number)
        new_df = pd.DataFrame({'#': number,
                               'layers': n_layers,
                               'neurons': n_neurons,
                               'TestMAE': mae,
                               'TestMSE': mse,
                               'TestLoss': loss,
                               '% Error': error}, index=[number])
        df = df.append(new_df)

        test_predictions = model.predict(normed_x_test)
        plot_pred(y_test, test_predictions, number)
        if loss < best_loss:
            best_loss = loss
            best_model = copy.copy(model)
            best_number = number
        del model
df.to_csv('./out.csv')
best_model.save('best_model_' + str(best_number) + '.h5')


