import tensorflow as tf
import torch as t

#keras
#lstm = L.LSTM(units=H, return_sequences=True, return_state=True)

#pytorch
#lstm = nn.LSTM(input_size=D, hidden_size=H, num_layers=1, batch_first=True).cuda()

def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )

def pytorch_LSTM(rnn_units, embedding_dim):
    return t.nn.LSTM(
        input_size=embedding_dim,
        hidden_size=rnn_units,
        num_layers=1,
        batch_first=True
    )

### Defining the RNN Model ###

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        # Layer 1: Embedding layer to transform indices into dense vectors
        #   of a fixed embedding size
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

        # Layer 2: LSTM with `rnn_units` number of units.
        LSTM(rnn_units),

        # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
        #   into the vocabulary size.
        tf.keras.layers.Dense(units=vocab_size)
    ])

    model_torch = t.nn.Sequential(
        t.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0),
        pytorch_LSTM(rnn_units, embedding_dim),
        t.nn.Linear(vocab_size, vocab_size)
    )

    return model

