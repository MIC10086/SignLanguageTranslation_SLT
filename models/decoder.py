import tensorflow as tf

def decoder_v1_0(input_word, lstm1_states, lstm2_states, rnn_units, embedding_units, 
    vocab_size, dropout=0.0, recurrent_dropout=0.0):
    #Input shape (b, n) [(b, r),(b, r)] [(b, r),(b, r)]

    x = tf.keras.layers.Embedding(vocab_size, embedding_units, mask_zero=True, 
        name='embedding_dec')(input_word)

    x = tf.keras.layers.LSTM(rnn_units, return_sequences=True, dropout=dropout, 
        recurrent_dropout=recurrent_dropout, name='lstm_1_dec')(x, initial_state=lstm1_states)
    x = tf.keras.layers.LSTM(rnn_units, return_sequences=True, dropout=dropout, 
        recurrent_dropout=recurrent_dropout, name='lstm_2_dec')(x, initial_state=lstm2_states)

    return x # Output shape (b, n, r)