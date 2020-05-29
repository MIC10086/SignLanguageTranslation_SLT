import tensorflow as tf

def encoder_v1_0(reduce_features, rnn_units, embedding_units, 
    dropout=0.0, recurrent_dropout=0.0):
    # Input shape (b, t, h*w)

    x = tf.keras.layers.Dense(units=embedding_units, name="dense_rf_enc")(reduce_features)

    x, h1, c1 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, name='lstm_1_enc')(x)

    x, h2, c2 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, name='lstm_2_enc')(x)

    return x, h1, c1, h2, c2 # Output shape (b, t, r) [(b, r),(b, r)] [(b, r),(b, r)]

def encoder_v1_1(reduce_features, rnn_units, embedding_units, 
    dropout=0.0, recurrent_dropout=0.0):
    # Input shape (b, t, h*w)

    x, h1, c1 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, name='lstm_1_enc')(reduce_features)

    x, h2, c2 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, name='lstm_2_enc')(x)

    return x, h1, c1, h2, c2 # Output shape (b, t, r) [(b, r),(b, r)] [(b, r),(b, r)]