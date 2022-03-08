import tensorflow as tf
import torch
from collections import OrderedDict

'''
kemb = tf.keras.layers.Embedding(83, 256, batch_input_shape=[4, None])
kinput = np.array([[randint(0,83) for j in range(100)] for i in range(4)])
ktest = emb(input)

pemb = t.nn.Embedding(4*100, 4*100*256)
pinput = 
ptest = 
'''


# keras
# lstm = L.LSTM(units=H, return_sequences=True, return_state=True)

# pytorch
# lstm = nn.LSTM(input_size=D, hidden_size=H, num_layers=1, batch_first=True).cuda()

def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )


def pytorch_LSTM(rnn_units, embedding_dim):
    return torch.nn.LSTM(
        input_size=embedding_dim,
        hidden_size=rnn_units,
        num_layers=1,
        batch_first=True
    )


### Defining the RNN Model ###
'''
vocab_size = 83
embedding_dim = 256
rnn_units = ?
batch_size = 4
'''
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
    '''
    model_torch = t.nn.Sequential(
        t.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0),
        pytorch_LSTM(rnn_units, embedding_dim),
        t.nn.Linear(vocab_size, vocab_size)
    )
    '''
    return model

class GetLSTMOutput(torch.nn.Module):
    def forward(self, x):
        out, _ = x
        return out

class MusicGenerator(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size, seq_length):
        super().__init__()

        self.lstm_model = torch.nn.Sequential(
            torch.nn.Embedding(batch_size*seq_length, embedding_dim),
            torch.nn.LSTM(embedding_dim, batch_size*embedding_dim, batch_first=True),
            GetLSTMOutput(),
            torch.nn.Linear(rnn_units, vocab_size)
        )

        '''
        self.lstm_model = torch.nn.Sequential(OrderedDict([
            ('embedding', torch.nn.Embedding(batch_size*seq_length, embedding_dim)),
            ('lstm', torch.nn.LSTM(embedding_dim, batch_size*embedding_dim, batch_first=True)),
            ('dense', torch.nn.Linear(rnn_units, vocab_size))
        ]))
        '''

    def forward(self, x):
        x = self.lstm_model(torch.tensor(x))
        print("Layer----------")
        return x
