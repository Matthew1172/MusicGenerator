import torch.distributions.distribution
from Transformer_Model import *
import data
import os
import time

if(torch.cuda.is_available()):
    print("GPU: ",torch.cuda.get_device_name(0), " is available, Switching now.")
else:
    print("GPU is not available, using CPU.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device is now: ", device)

#size of word embeddings
emsize = 512
#number of hidden units per layer
hidden_units = 256
#number of layers
nlayers = 256
#initial learning rate
learning_rate = 1e-1
#gradient clipping
clip = 25e-2
#upper epoch limit
epochs = 40
#batch size
batch_size = 20
#sequence length
bptt = 35
#dropout applied to layers (0 = no dropout)
dropout = 2e-1
#report interval
log_interval = 200
#the number of heads in the encoder/decoder of the transformer model
num_heads = 8

cwd = os.getcwd()
#dataset = "./dataset/irish"
#dataset = "./dataset/australian"
dataset = "./dataset/set1"

# Checkpoint location:
CHECKPOINT_DIR = 'training_checkpoints_pytorch'

try:
    os.mkdir(CHECKPOINT_DIR)
except FileExistsError:
    print("The directory {} already exists...".format(CHECKPOINT_DIR))

CHECKPOINT_PREFIX = 'my_ckpt.pth'

CHECKPOINT_DIR = os.path.join(cwd, CHECKPOINT_DIR)
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)

myCorpus = data.Corpus(dataset)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(myCorpus.train, batch_size)
val_data = batchify(myCorpus.valid, eval_batch_size)
test_data = batchify(myCorpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(myCorpus.dictionary)
model = TransformerModel(ntokens, emsize, num_heads, hidden_units, nlayers, device, device, dropout).to(device)

#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(myCorpus.dictionary)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(myCorpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        #model.zero_grad()
        optimizer.zero_grad()
        output = model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        '''
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)
        '''

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = learning_rate
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(CHECKPOINT_PREFIX, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            #lr /= 4.0
            pass
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(CHECKPOINT_PREFIX, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)