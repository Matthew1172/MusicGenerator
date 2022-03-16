import numpy as np

import torch.distributions.distribution
from tqdm import tqdm

#multi processing imports
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from MySong import *
from LSTM_Model import *


# Optimization parameters:
num_training_iterations = 15000  # Increase this to train longer
batch_size = 64  # Experiment between 1 and 64
seq_length = 500  # Experiment between 50 and 500
learning_rate = 25e-3  # Experiment between 1e-5 and 1e-1

# Model parameters:
embedding_dim = 256
rnn_units = 2048  # Experiment between 1 and 2048
epochs = 1


# Checkpoint location:
checkpoint_dir = 'training_checkpoints_pytorch'
checkpoint_prefix = 'my_ckpt.pth'
checkpoint_dir = os.path.join(cwd, checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, checkpoint_prefix)

songs = []
with open(os.path.join(cwd, 'dataset', 'irish.abc'), 'r') as f:
    text = f.read()
    songs = extract_song_snippet(text)
# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs)
# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")

vocab_size = len(vocab)
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def vectorize_string(string):
    vectorized_list = np.array([char2idx[s] for s in string])
    return vectorized_list

vectorized_songs = vectorize_string(songs_joined)

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









class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)





    model = MyLSTM(vocab_size, embedding_dim, rnn_units, batch_size, seq_length)
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9)

    CHECKPOINT_PATH = checkpoint_prefix

    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)


    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))


    history = []
    if hasattr(tqdm, '_instances'): tqdm._instances.clear()
    for epoch in range(epochs):
        hn = torch.zeros(1, 1, rnn_units).to(rank)  # [num_layers*num_directions,batch,hidden_size]
        cn = torch.zeros(1, 1, rnn_units).to(rank)  # [num_layers*num_directions,batch,hidden_size]

        for iter in tqdm(range(num_training_iterations)):

            # Grab a batch and propagate it through the network
            x, y = get_batch(vectorized_songs, seq_length, batch_size)


            #loss, (hn, cn) = torch_train(x_batch, y_batch, hn, cn)
            x = torch.tensor(x).to(rank)
            optimizer.zero_grad()
            y_hat, (hn, cn) = ddp_model(x, hn, cn)
            y_hat.to(rank)
            hn.to(rank)
            cn.to(rank)
            #loss = compute_loss(y, y_hat)
            a = y_hat.permute((0, 2, 1))  # shape of preds must be (N, C, H, W) instead of (N, H, W, C)
            a.to(rank)
            b = torch.tensor(y, device=rank).long()  # shape of labels must be (N, H, W) and type must be long integer
            F = torch.nn.CrossEntropyLoss()
            loss = F(a, b)
            loss.to(rank)

            # compute gradients (grad)
            loss.backward()
            optimizer.step()



            # Update the progress bar
            history.append(loss.cpu().detach().numpy().mean())
            #plotter.plot(history)

            # Update the model with the changed weights!
            if iter % 100 == 0 and rank == 0:
                torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    if rank == 0:
        # Save the trained model and the weights
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    cleanup()


def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = (rank * 2) % world_size
    dev1 = (rank * 2 + 1) % world_size
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    #run_demo(demo_basic, world_size)
    run_demo(demo_checkpoint, world_size)
    #run_demo(demo_model_parallel, world_size)