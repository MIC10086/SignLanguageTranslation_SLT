import tensorflow as tf

def real_acc(real, pred):
    """ Real accuracy by batch, not for epoch"""
    pred_indexes = tf.argmax(pred, axis=-1)
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=pred_indexes.dtype)
    
    pred_indexes = tf.multiply(pred_indexes, mask)
    
    equals = tf.math.equal(tf.cast(real, pred_indexes.dtype), pred_indexes)
    return tf.math.reduce_mean(tf.cast(equals, tf.float32))