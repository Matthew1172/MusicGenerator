import numpy as np
import os
import time
import regex as re
import subprocess
import urllib
import functools

import torch.distributions.distribution
from IPython import display as ipythondisplay
from tqdm import tqdm
import matplotlib.pyplot as plt

from MySong import *
from Graph import PeriodicPlotter
from LSTM_Model import *

if(torch.cuda.is_available()):
    print("GPU: ",torch.cuda.get_device_name(0)," is available, Switching now.")
else:
    print("GPU is not available, using CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is now: ", device)

train = True
inference = True
### Hyperparameter setting and optimization ###

epochs = 1

# Optimization parameters:
num_training_iterations = 2000  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 1e-1  # Experiment between 1e-5 and 1e-1

# Model parameters:
embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location:
checkpoint_dir = 'training_checkpoints_pytorch'
checkpoint_prefix = 'my_ckpt.pth'

checkpoint_dir = os.path.join(cwd, checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, checkpoint_prefix)


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

for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
    print("Step {:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


# Build a simple model with default hyperparameters. You will get the
#   chance to change these later.
#model = MusicGenerator(len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=batch_size, seq_length=seq_length)

model = MyLSTM(vocab_size, embedding_dim, rnn_units, batch_size, seq_length)
model.to(device)
print(model)

x, y = get_batch(vectorized_songs, seq_length=100, batch_size=4)

hn = torch.zeros(1, 1, rnn_units).to(device)  # [num_layers*num_directions,batch,hidden_size]
cn = torch.zeros(1, 1, rnn_units).to(device)  # [num_layers*num_directions,batch,hidden_size]
x = torch.tensor(x).to(device)
pred, (hn, cn) = model(x, hn, cn)
#pred = model(x)

print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = torch.distributions.categorical.Categorical(logits=pred[0]).sample()
sampled_indices = torch.squeeze(sampled_indices, dim=-1).cpu().numpy()

print("Input: \n", repr("".join(idx2char[x.cpu()[0]])))
# print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

### Defining the loss function ###

def compute_loss(labels, logits):
    x = logits.permute((0, 2, 1))  # shape of preds must be (N, C, H, W) instead of (N, H, W, C)
    x.to(device)
    y = torch.tensor(labels, device=device).long()  # shape of labels must be (N, H, W) and type must be long integer
    F = torch.nn.CrossEntropyLoss()
    loss = F(x, y)
    loss.to(device)
    return loss

example_batch_loss = compute_loss(y, pred)

print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)")
'''
example_batch_loss.numpy().mean() = 4.417909
'''
print("scalar_loss:      ", example_batch_loss.cpu().detach().numpy().mean())
print(example_batch_loss)




### Define optimizer and training operation ###

model = MyLSTM(vocab_size, embedding_dim, rnn_units, batch_size, seq_length)
model.to(device)

#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

def torch_train(x, y, hn, cn):
    x = torch.tensor(x).to(device)
    optimizer.zero_grad()
    # forward pass and loss calculation
    # implicit tape-based AD
    y_hat, (hn, cn) = model(x, hn, cn)
    y_hat.to(device)
    hn.to(device)
    cn.to(device)
    loss = compute_loss(y, y_hat)

    # compute gradients (grad)
    loss.backward()
    optimizer.step()
    return loss, (hn, cn)

if train:

    # try to create pytorch training checkpoints directory
    try:
        os.mkdir(checkpoint_dir)
    except FileExistsError:
        print("The pytorch training directory already exists...")

    ##################
    # Begin training!#
    ##################
    history = []
    #plotter = PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists

    for epoch in range(epochs):
        hn = torch.zeros(1, 1, rnn_units).to(device)  # [num_layers*num_directions,batch,hidden_size]
        cn = torch.zeros(1, 1, rnn_units).to(device)  # [num_layers*num_directions,batch,hidden_size]

        for iter in tqdm(range(num_training_iterations)):

            # Grab a batch and propagate it through the network
            x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
            loss, (hn, cn) = torch_train(x_batch, y_batch, hn, cn)

            # Update the progress bar
            history.append(loss.cpu().detach().numpy().mean())
            #plotter.plot(history)

            # Update the model with the changed weights!
            if iter % 100 == 0:
                torch.save(model.state_dict(), checkpoint_prefix)

    # Save the trained model and the weights
    torch.save(model.state_dict(), checkpoint_prefix)


if(inference):

    ### Prediction of a generated song ###

    def generate_text(model, start_string, generation_length=1000):
        # Evaluation step (generating ABC text using the learned RNN model)

        input_eval = [char2idx[s] for s in start_string]
        input_eval = np.expand_dims(input_eval, axis=0)

        # Empty string to store our results
        text_generated = []

        # Here batch size == 1
        '''
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        '''
        hn = torch.zeros(1, 1, rnn_units).to(device)  # [num_layers*num_directions,batch,hidden_size]
        cn = torch.zeros(1, 1, rnn_units).to(device)  # [num_layers*num_directions,batch,hidden_size]

        tqdm._instances.clear()

        for i in tqdm(range(generation_length)):
            input_eval = torch.tensor(input_eval).to(device)
            predictions, (hn, cn) = model(input_eval, hn, cn)
            predictions.to(device)

            # Remove the batch dimension
            # predictions = tf.squeeze(predictions, 0)
            predictions = torch.squeeze(predictions, dim=0)

            # predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            predicted_id = torch.distributions.categorical.Categorical(logits=predictions).sample()[0].cpu().numpy()
            # predicted_id = torch.distributions.categorical.Categorical(logits=predictions)

            # Pass the prediction along with the previous hidden state
            #   as the next inputs to the model
            input_eval = np.expand_dims(np.array([predicted_id]), axis=0)

            '''add the predicted character to the generated text!'''
            # Hint: consider what format the prediction is in vs. the output
            text_generated.append(idx2char[predicted_id])

        return (start_string + ''.join(text_generated))

    # Restore the model weights for the last checkpoint after training
    model = MyLSTM(vocab_size, embedding_dim, rnn_units, batch_size, seq_length)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_prefix))
    model.eval()
    print(model)

    '''Use the model and the function defined above to generate ABC format text of length 1000!
        As you may notice, ABC files start with "X" - this may be a good start string.'''
    generated_text = generate_text(model, start_string="X", generation_length=1000)

    generated_songs = extract_song_snippet(generated_text)

    for i, song in enumerate(generated_songs):
        # could be incorrect ABC notational syntax, save the ABC file anyway...
        print("---------------------------------------------------------------")
        print("Generated song", i)
        n = "gan_song_{}".format(i)
        basename = os.path.join(op, save_song_to_abc(song, filename=n))
        abc2wav(basename + '.abc')
