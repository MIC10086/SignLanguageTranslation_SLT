import tensorflow as tf

def output_v1_0(decoder_output, sta_output, vocab_size):
    # Input shape (b, n, r) (b, n, h*w)

    x = tf.concat([decoder_output, sta_output], axis=-1, name='concat_out')

    x = tf.keras.layers.Dense(units=vocab_size, activation='softmax', name='prediction_out')(x)

    return x # Output shape (b, vocab size)