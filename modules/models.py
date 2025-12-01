import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from sklearn.ensemble import RandomForestRegressor

def build_snn(input_dim, p):
    L2 = regularizers.l2(p["l2_dense"])
    m = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(p["units"], activation="relu", kernel_regularizer=L2),
        layers.Dense(1, activation="relu")
    ])
    m.compile(optimizer=optimizers.Adam(p["lr"]), loss=tf.keras.losses.Huber())
    return m


def build_dnn(input_dim, p):
    L2 = regularizers.l2(p["l2_dense"])
    m = models.Sequential([layers.Input(shape=(input_dim,))])
    for _ in range(p["layers"]):
        m.add(layers.Dense(p["units"], activation="relu", kernel_regularizer=L2))
        if p["dropout"] > 0:
            m.add(layers.Dropout(p["dropout"]))
    m.add(layers.Dense(1, activation="relu"))
    m.compile(optimizer=optimizers.Adam(p["lr"]), loss=tf.keras.losses.Huber())
    return m


def build_cnn(input_shape, p):
    L2c = regularizers.l2(p["l2_conv"])
    L2d = regularizers.l2(p["l2_dense"])

    m = models.Sequential()
    m.add(layers.Input(shape=input_shape))

    # dynamic pooling
    safe_layers = 0
    h = input_shape[0]
    for _ in range(p["conv_layers"]):
        if h // 2 >= 2:
            safe_layers += 1
            h //= 2
        else:
            break

    for _ in range(safe_layers):
        m.add(layers.Conv2D(
            p["filters"],
            (p["kernel_size"], p["kernel_size"]),
            padding="same",
            activation="relu",
            kernel_regularizer=L2c
        ))
        if p["dropout"] > 0:
            m.add(layers.Dropout(p["dropout"]))
        m.add(layers.MaxPooling2D((2, 2)))

    if safe_layers < p["conv_layers"]:
        m.add(layers.Conv2D(
            p["filters"],
            (p["kernel_size"], p["kernel_size"]),
            padding="same",
            activation="relu",
            kernel_regularizer=L2c
        ))

    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(256, activation="relu", kernel_regularizer=L2d))
    if p["dropout"] > 0:
        m.add(layers.Dropout(p["dropout"]))
    m.add(layers.Dense(1, activation="relu"))

    m.compile(optimizer=optimizers.Adam(p["lr"]), loss=tf.keras.losses.Huber())
    return m


def build_rf(p):
    return RandomForestRegressor(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"],
        min_samples_leaf=p["min_samples_leaf"],
        max_features=p["max_features"],
        n_jobs=-1,
        random_state=42
    )
