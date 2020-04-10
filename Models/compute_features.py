import tensorflow as tf

def compute_features_v1_0(input_video):
    # Input video shape (b, t, h, w, c)
    
    # Conv1
    x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, padding="same", activation="relu",
                              name='conv3d_cf_1')(input_video)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2), name='max_pool3d_1')(x)

    # Conv2
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="same", activation="relu",
                          name='conv3d_cf_2')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2), name='max_pool3d_cf_2')(x)

    # Conv3
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu",
                          name='conv3d_cf_3')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2),name='max_pool3d_cf_3')(x)

    # Conv4
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="same", activation="relu",
                          name='conv3d_cf_4')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2),name='max_pool3d_cf_4')(x)

    return x # Output shape (b, t, h/16, w/16, 128)