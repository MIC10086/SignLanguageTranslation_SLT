import tensorflow as tf
from utils.load_weights import loadweights

def compute_features_v1_0(input_video, pretrained=(None, None), weight_decay=None):
    """pretrained is a tuple with (model, path_to_weights)"""
    # Input video shape (b, t, h, w, c)

    weights = loadweights(pretrained[0], pretrained[1])

    # Conv1
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=weight_decay, weights=weights[0], name='conv3d_cf_1')(input_video)
    x = tf.keras.layers.BatchNormalization(axis=4, name="batch_norm_cf_1")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2), name='max_pool3d_cf_1')(x)

    # Conv2
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=weight_decay, weights=weights[1], name='conv3d_cf_2')(x)
    x = tf.keras.layers.BatchNormalization(axis=4, name="batch_norm_cf_2")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2), name='max_pool3d_cf_2')(x)

    # Conv3
    x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=weight_decay, weights=weights[2], name='conv3d_cf_3')(x)
    x = tf.keras.layers.BatchNormalization(axis=4, name="batch_norm_cf_3")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2),name='max_pool3d_cf_3')(x)

    # Conv4
    x = tf.keras.layers.Conv3D(filters=512, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=weight_decay, weights=weights[3], name='conv3d_cf_4')(x)
    x = tf.keras.layers.BatchNormalization(axis=4, name="batch_norm_cf_4")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,1,1),name='max_pool3d_cf_4')(x)

    return x # Output shape (b, t/2, h/8-1, w/8-1, 128)

def compute_features_v1_1(input_video, pretrained=None, weight_decay=None):
    # Input video shape (b, t, h, w, c)

    if pretrained:
        raise NotImplementedError('Use pretrained weights with v1.1 is not implemented yet...')
    
    # Conv1
    x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=weight_decay, name='conv3d_cf_1')(input_video)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2), name='max_pool3d_cf_1')(x)

    # Conv2
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=weight_decay, name='conv3d_cf_2')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2), name='max_pool3d_cf_2')(x)

    # Conv3
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=weight_decay, name='conv3d_cf_3')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2),name='max_pool3d_cf_3')(x)

    # Conv4
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding="same", activation="relu",
        kernel_regularizer=weight_decay, name='conv3d_cf_4')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2),name='max_pool3d_cf_4')(x)

    return x # Output shape (b, t, h/16, w/16, 128)