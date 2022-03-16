import torch

### Defining the RNN Model ###
class MyLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size, seq_length, dev0, dev1):
        super(MyLSTM, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.embedding = torch.nn.Embedding(batch_size*seq_length, embedding_dim).to(dev0)
        self.lstm = torch.nn.LSTM(batch_first=True, input_size=embedding_dim, hidden_size=rnn_units).to(dev1)
        self.linear = torch.nn.Linear(in_features=rnn_units, out_features=vocab_size).to(dev0)

    def forward(self, x, hn, cn):
        x = x.to(self.dev0)
        hn = hn.to(self.dev1)
        cn = cn.to(self.dev1)

        embeds = self.embedding(x)
        embeds = embeds.to(self.dev0)
        # Stateful
        embeds_longer = embeds.view(1,embeds.shape[0]*embeds.shape[1], embeds.shape[2])
        embeds_longer = embeds_longer.to(self.dev1)
        out_longer, (hn, cn) = self.lstm(embeds_longer, (hn.detach(), cn.detach()))
        hn = hn.to(self.dev0)
        cn = cn.to(self.dev0)
        out_longer = out_longer.to(self.dev0)
        hn = hn.to(self.dev0)
        cn = cn.to(self.dev0)
        out = out_longer.view(embeds.shape[0],embeds.shape[1],out_longer.shape[2])
        out = out.to(self.dev0)
        out = self.linear(out)
        out = out.to(self.dev0)
        return out, (hn, cn)