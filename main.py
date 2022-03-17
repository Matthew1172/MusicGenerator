import random

import numpy as np
import torch.distributions.distribution
from tqdm import tqdm
from MySong import *
from Transformer_Model import *

if(torch.cuda.is_available()):
    print("GPU: ",torch.cuda.get_device_name(1), " is available, Switching now.")
else:
    print("GPU is not available, using CPU.")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device is now: ", device)

#gen_length = 1000
#logging interval
log_interval = 200
### Hyperparameter setting and optimization ###

epochs = 400

# Optimization parameters:
#num_training_iterations = 4000
batch_size = 100  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-2  # Experiment between 1e-5 and 1e-1

# Model parameters:
embedding_dim = 256
#hidden units per layer
hidden_units = 400
#number of layers
nlayers = 128
num_heads = 8
#num_encoder_layers = 3
#num_decoder_layers = 3
dropout = 2e-1
#gradient clipping
clip = 25e-2

# Checkpoint location:
CHECKPOINT_DIR = 'training_checkpoints_pytorch'
CHECKPOINT_PREFIX = 'my_ckpt.pth'

CHECKPOINT_DIR = os.path.join(cwd, CHECKPOINT_DIR)
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)


songs = []
with open(os.path.join(cwd, 'dataset', 'irish.abc'), 'r') as f:
    text = f.read()
    songs = extract_song_snippet(text)

# Print one of the songs to inspect it in greater detail!
example_song = songs[0]
print("\nExample song: ")
print(example_song)

# need abc2midi and timidity
play_song(example_song)

# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs)

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")

vocab_size = len(vocab)

### Define numerical representation of text ###

# Create a mapping from character to unique index.
# For example, to get the index of the character "d",
#   we can evaluate `char2idx["d"]`.
char2idx = {u: i for i, u in enumerate(vocab)}

# Create a mapping from indices to characters. This is
#   the inverse of char2idx and allows us to convert back
#   from unique index to the character in our vocabulary.
idx2char = np.array(vocab)  ### Define numerical representation of text ###

# Create a mapping from character to unique index.
# For example, to get the index of the character "d",
#   we can evaluate `char2idx["d"]`.
char2idx = {u: i for i, u in enumerate(vocab)}

# Create a mapping from indices to characters. This is
#   the inverse of char2idx and allows us to convert back
#   from unique index to the character in our vocabulary.
idx2char = np.array(vocab)

print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

### Vectorize the songs string ###

'''
  NOTE: the output of the `vectorize_string` function 
  should be a np.array with `N` elements, where `N` is
  the number of characters in the input string
'''


def vectorize_string(string):
    vectorized_list = np.array([char2idx[s] for s in string])
    return vectorized_list


vectorized_songs = vectorize_string(songs_joined)

print('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
# check that vectorized_songs is a numpy array
assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"


def test_batch_func_types(func, args):
    ret = func(*args)
    assert len(ret) == 2, "[FAIL] get_batch must return two arguments (input and label)"
    assert type(ret[0]) == np.ndarray, "[FAIL] test_batch_func_types: x is not np.array"
    assert type(ret[1]) == np.ndarray, "[FAIL] test_batch_func_types: y is not np.array"
    print("[PASS] test_batch_func_types")
    return True


def test_batch_func_shapes(func, args):
    dataset, seq_length, batch_size = args
    x, y = func(*args)
    correct = (batch_size, seq_length)
    assert x.shape == correct, "[FAIL] test_batch_func_shapes: x {} is not correct shape {}".format(x.shape, correct)
    assert y.shape == correct, "[FAIL] test_batch_func_shapes: y {} is not correct shape {}".format(y.shape, correct)
    print("[PASS] test_batch_func_shapes")
    return True


def test_batch_func_next_step(func, args):
    x, y = func(*args)
    assert (x[:, 1:] == y[:, :-1]).all(), "[FAIL] test_batch_func_next_step: x_{t} must equal y_{t-1} for all t"
    print("[PASS] test_batch_func_next_step")
    return True


### Batch definition to create training examples ###

def get_batch(vectorized_songs, seq_length, batch_size):
    # the length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n - seq_length, batch_size)

    input_batch = [vectorized_songs[i:i + seq_length] for i in idx]
    output_batch = [vectorized_songs[i + 1: i + 1 + seq_length] for i in idx]

    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch


# Perform some simple tests to make sure your batch function is working properly!
test_args = (vectorized_songs, 10, 2)
if not test_batch_func_types(get_batch, test_args) or \
        not test_batch_func_shapes(get_batch, test_args) or \
        not test_batch_func_next_step(get_batch, test_args):
    print("======\n[FAIL] could not pass tests")
else:
    print("======\n[PASS] passed all tests!")

x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

vectorized_songs = torch.tensor(vectorized_songs, device=device)

train_data = batchify(vectorized_songs, batch_size)
val_data = batchify(vectorized_songs, batch_size)
test_data = batchify(vectorized_songs, batch_size)


for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
    print("Step {:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


model = TransformerModel(vocab_size, embedding_dim, num_heads, hidden_units, nlayers, dropout).to(device)
print(model)
#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch2(source, i):
    seq = min(seq_length, len(source) - 1 - i)
    data = source[i:i+seq]
    target = source[i+1:i+1+seq].view(-1)
    return data, target

#pass batches in as (batch_size, seq_length)
def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = vocab_size
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_length):
            data, targets = get_batch2(data_source, i)
            '''TODO: squeeze data and targets'''
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets.type(torch.int64)).item()
    return total_loss / (len(data_source) - 1)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = vocab_size
    for batch, i in enumerate(range(0, vectorized_songs.size(0) - 1, seq_length)):
        data, targets = get_batch2(train_data, i)
        '''TODO: squeeze data and targets'''
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output = model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets.type(torch.int64))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-learning_rate)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // seq_length, learning_rate,
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
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)