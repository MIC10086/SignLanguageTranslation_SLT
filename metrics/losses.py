#File for losses to use in the model
import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

def SparseCategoricalCrossentropy_mask(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss = tf.multiply(loss_, mask)

    return tf.reduce_mean(loss)