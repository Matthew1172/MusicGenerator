import torch

'''
kemb = tf.keras.layers.Embedding(83, 256, batch_input_shape=[4, None])
kinput = np.array([[randint(0,83) for j in range(100)] for i in range(4)])
ktest = emb(input)

pemb = t.nn.Embedding(4*100, 4*100*256)
pinput = 
ptest = 
'''

### Defining the RNN Model ###
'''
vocab_size = 83
embedding_dim = 256
rnn_units = 1024
batch_size = 4
'''
class GetLSTMOutput(torch.nn.Module):
    def forward(self, x):
        out, _ = x
        return out

class MusicGenerator(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size, seq_length):
        super().__init__()

        self.lstm_model = torch.nn.Sequential(
            torch.nn.Embedding(batch_size*seq_length, embedding_dim),
            torch.nn.LSTM(embedding_dim, rnn_units, batch_first=True),
            GetLSTMOutput(),
            torch.nn.Linear(rnn_units, vocab_size)
        )

    def forward(self, x):
        x = self.lstm_model(torch.tensor(x))
        return x
