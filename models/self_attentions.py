import tensorflow as tf

def self_attention_v1_0(encoder_output):
    # Input shape (b, t, r)

    q = tf.keras.layers.Dense(units=encoder_output.shape[2], use_bias=False,
        name="dense_q_sea")(encoder_output)
    k = tf.keras.layers.Dense(units=encoder_output.shape[2], use_bias=False,
        name="dense_k_sea")(encoder_output)
    v = tf.keras.layers.Dense(units=encoder_output.shape[2], use_bias=False,
        name="dense_v_sea")(encoder_output)

    x = tf.matmul(a=q, b=k, transpose_b=True, name="matmul_qk_sea")

    r = tf.shape(encoder_output, name="get_shape_sea")[-1]
    r = tf.cast(r, tf.float32, name="cast_shape_sea")
    r = tf.sqrt(r, name="sqrt_r_sea")

    x = tf.divide(x = x, y = r, name = "divide_qk_r_sea")

    x = tf.keras.layers.Softmax(axis=-1, name = 'att_weights_sea')(x)

    x = tf.matmul(a=x, b=v, name="context_vector_sea")

    return x