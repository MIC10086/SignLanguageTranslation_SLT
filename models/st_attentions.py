import tensorflow as tf

def st_attention_v1_0(decoder_output, reduce_features):
    # Input shapes (b, n, r) (b, t, h*w)
    
    q = tf.tile(decoder_output, [1,reduce_features.shape[1],1], name="repeat_q_sta")
    q = tf.keras.layers.Reshape(target_shape=[reduce_features.shape[1], 
                                              decoder_output.shape[1],
                                              decoder_output.shape[2]], 
        name="reshape_q_sta")(q)
    q = tf.keras.layers.Dense(units=reduce_features.shape[2], use_bias=False,
        name="dense_q_sta")(q)
    
    v = tf.tile(reduce_features, [1,decoder_output.shape[1],1], name="repeat_v_sta")
    v = tf.keras.layers.Reshape(target_shape=[decoder_output.shape[1], 
                                              reduce_features.shape[1],
                                              reduce_features.shape[2]], 
        name="reshape_v_sta")(v)
    v = tf.transpose(v, perm=[0,2,1,3], name='transpose_v_sta')

    x = tf.matmul(a=q, b=v, transpose_b=True, name="matmul_qv_sta")

    x = tf.matmul(a=x, b=v, name="matmul_qvv_sta")

    x = tf.transpose(x, perm=[0,2,1,3], name='transpose_qvv_sta')

    x = tf.keras.layers.Softmax(axis=-1, name='att_weights_sta')(x)

    x = tf.multiply(x, reduce_features, name='mul_pw_qvv_sta')

    x = tf.reduce_sum(x, axis=-2, name='context_vector_sta')

    return x # Output shape (b, n, h*w)

def st_attention_v1_4_0(decoder_output, encoder_output, reduce_features):
    # Input shapes (b, n, r) (b, t, r) (b, t, h*w)
    
    x = tf.matmul(a=decoder_output, b=encoder_output, transpose_b=True, name='matmul_qk_sta')

    x = tf.expand_dims(x, axis=-1, name='expand_dims_qk_sta')

    x = tf.keras.layers.Dense(units=reduce_features.shape[2], use_bias=False,
        name="dense_qk_sta")(x)

    red_feat = tf.expand_dims(reduce_features, axis=1, name='expand_dims_v_sta')

    x = tf.multiply(x, red_feat, name='mul_pw_qkv_sta')

    x = tf.keras.layers.Reshape(target_shape=[decoder_output.shape[1], -1], 
        name="reshape_qkv_sta")(x)

    x = tf.keras.layers.Softmax(axis=-1, name='att_weights_sta')(x)

    v_flat = tf.keras.layers.Flatten(name='flat_v_sta')(reduce_features)

    v_flat_exp = tf.expand_dims(v_flat, axis=1, name="expand_dims_vflat_sta")

    x = tf.multiply(x, v_flat_exp, name='mul_pw_qkvv_sta')

    x = tf.keras.layers.Reshape(target_shape=[decoder_output.shape[1],
                                              reduce_features.shape[1],
                                              reduce_features.shape[2]],
        name='reshape_qkvv_sta')(x)

    x = tf.reduce_sum(x, axis=-2, name='context_vector_sta')

    return x # Output shape (b, n, h*w)

def st_attention_v1_4_1(decoder_output, encoder_output, reduce_features):
    # Input shapes (b, n, r) (b, t, r) (b, t, h*w)
    
    x = tf.matmul(a=decoder_output, b=encoder_output, transpose_b=True, name='matmul_qk_sta')

    x = tf.expand_dims(x, axis=-1, name='expand_dims_qk_sta')

    x = tf.keras.layers.Dense(units=reduce_features.shape[2], use_bias=False,
        name="dense_qk_sta")(x)

    red_feat = tf.expand_dims(reduce_features, axis=1, name='expand_dims_v_sta')

    x = tf.multiply(x, red_feat, name='mul_pw_qkv_sta')

    x = tf.keras.layers.Softmax(axis=-1, name='att_weights_sta')(x)

    x = tf.multiply(x, red_feat, name='mul_pw_qkvv_sta')

    x = tf.reduce_sum(x, axis=-2, name='context_vector_sta')

    return x # Output shape (b, n, h*w)

def st_attention_v1_5(decoder_output, encoder_output, reduce_features):
    # Input shapes (b, n, r) (b, t, r) (b, t, h*w)

    x = tf.matmul(a=decoder_output, b=encoder_output, transpose_b=True, name='matmul_qk_sta')

    x = tf.keras.layers.Dense(units=reduce_features.shape[1]*reduce_features.shape[2], 
        use_bias=False, name="dense_qk_sta")(x)

    v_flat = tf.keras.layers.Flatten(name='flat_v_sta')(reduce_features)

    v_flat_exp = tf.expand_dims(v_flat, axis=1, name="expand_dims_vflat_sta")

    x = tf.multiply(x, v_flat_exp, name='mul_pw_qkv_sta')

    x = tf.keras.layers.Softmax(axis=-1, name='att_weights_sta')(x)

    x = tf.multiply(x, v_flat_exp, name='mul_pw_qkvv_sta')

    x = tf.keras.layers.Reshape(target_shape=[decoder_output.shape[1],
                                              reduce_features.shape[1],
                                              reduce_features.shape[2]],
        name='reshape_qkvv_sta')(x)

    x = tf.reduce_sum(x, axis=-2, name='context_vector_sta')

    return x # Output shape (b, n, h*w)