import torch

### Defining the RNN Model ###
class MyLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size, seq_length):
        super().__init__()
        self.embedding = torch.nn.Embedding(batch_size*seq_length, embedding_dim)
        self.lstm = torch.nn.LSTM(batch_first=True, input_size=embedding_dim, hidden_size=rnn_units)
        self.linear = torch.nn.Linear(in_features=rnn_units, out_features=vocab_size)

    def forward(self, x, hn, cn):
        embeds = self.embedding(x)
        # Stateful
        embeds_longer = embeds.view(1,embeds.shape[0]*embeds.shape[1], embeds.shape[2])
        out_longer, (hn, cn) = self.lstm(embeds_longer, (hn.detach(), cn.detach()))
        out = out_longer.view(embeds.shape[0],embeds.shape[1],out_longer.shape[2])
        out = self.linear(out)
        return out, (hn, cn)