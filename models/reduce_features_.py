import tensorflow as tf

def reduce_features_v1_0(input_features):
    # Input features shape (b, t, h, w, c)

    x = tf.expand_dims(input_features, axis = -1, name="expand_dims_rf")

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv3D(filters=1, kernel_size=(1,1,input_features.shape[4]),
            name = "conv3d_rf"),
        name = "time_dist_rf"
    )(x)

    x = tf.keras.layers.Reshape(target_shape=[input_features.shape[1], -1], name="reshape_rf")(x)

    return x # Output shape (b, t, h*w)

def reduce_features_v1_1(input_features):
    # Input features shape (b, t, h, w, c)

    x = tf.keras.layers.Conv3D(filters=1, kernel_size=1, padding="same", activation="relu",
                              name='conv3d_rf')(input_features)
    
    x = tf.keras.layers.Reshape(target_shape=[input_features.shape[1], -1], name="reshape_rf")(x)

    return x # Output shape (b, t, h*w)

def reduce_features_v1_2(input_features):
    # Input features shape (b, t, h, w, c)

    x1 = tf.keras.layers.Conv3D(filters=1, kernel_size=3, padding="same", activation="relu",
                              name='conv3d_3x3_rf')(input_features)
    x2 = tf.keras.layers.Conv3D(filters=1, kernel_size=5, padding="same", activation="relu",
                              name='conv3d_5x5_rf')(input_features)
    x3 = tf.keras.layers.Conv3D(filters=1, kernel_size=7, padding="same", activation="relu",
                              name='conv3d_7x7_rf')(input_features)

    x = tf.concat([x1, x2, x3], axis=-1, name="concat_rf")

    x = tf.keras.layers.Conv3D(filters=1, kernel_size=1, padding="same", activation="relu",
                              name='conv3d_1x1_rf')(x)
    
    x = tf.keras.layers.Reshape(target_shape=[input_features.shape[1], -1], name="reshape_rf")(x)

    return x # Output shape (b, t, h*w)