import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor


def create_cnn(input_shape, l2=1e-5):
    """
    Build and compile a CNN for regression.

    Args:
        input_shape (tuple): (height, width, channels)
        l2 (float): L2 regularization factor

    Returns:
        tf.keras.Model: compiled CNN model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(128, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def create_dnn(input_dim, l2=1e-4):
    """
    Build and compile a deep feed-forward neural network.

    Args:
        input_dim (int): number of input features
        l2 (float): L2 regularization factor

    Returns:
        tf.keras.Model: compiled DNN model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dense(256, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def create_snn(input_dim, l2=1e-4):
    """
    Build and compile a single-hidden-layer neural network.

    Args:
        input_dim (int): number of input features
        l2 (float): L2 regularization factor

    Returns:
        tf.keras.Model: compiled SNN model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def create_rf(**kwargs):
    """
    Instantiate a RandomForestRegressor for regression.

    Args:
        **kwargs: RandomForest parameters (e.g., n_estimators)

    Returns:
        RandomForestRegressor: untrained RF model
    """
    params = {
        'n_estimators': 500,
        'max_depth': 10,
        'max_features': 'sqrt',
        'min_samples_leaf': 2,
        'min_samples_split': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    params.update(kwargs)
    return RandomForestRegressor(**params)
