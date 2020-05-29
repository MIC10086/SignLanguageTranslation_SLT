import tensorflow as tf

def decoder_v1_0(input_word, lstm1_h, lstm1_c, lstm2_h, lstm2_c, rnn_units, embedding_units, 
    vocab_size, dropout=0.0, recurrent_dropout=0.0):
    #Input shape (b, 1) [(b, r),(b, r)] [(b, r),(b, r)]

    x = tf.keras.layers.Embedding(vocab_size, embedding_units, mask_zero=True, 
        name='embedding_dec')(input_word)

    x, h1, c1 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, name='lstm_1_dec')(x, 
        initial_state=[lstm1_h, lstm1_c])
    x, h2, c2 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, 
        dropout=dropout, recurrent_dropout=recurrent_dropout, name='lstm_2_dec')(x, 
        initial_state=[lstm2_h, lstm2_c])

    return x, h1, c1, h2, c2 # Output shape (b, 1, r) [(b, r),(b, r)] [(b, r),(b, r)]