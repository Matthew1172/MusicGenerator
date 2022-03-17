###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model.
#
###############################################################################
import argparse

import numpy as np
import torch
import os
from MySong import *

seed = 49
temp = 1.0
gen_length = 1000
log_interval = 200

# Checkpoint location:
CHECKPOINT_DIR = 'training_checkpoints_pytorch'
CHECKPOINT_PREFIX = 'my_ckpt.pth'

cwd = os.getcwd()
CHECKPOINT_DIR = os.path.join(cwd, CHECKPOINT_DIR)
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)
GENERATION_PREFIX = os.path.join(cwd, "generated.txt")

# Set the random seed manually for reproducibility.
torch.manual_seed(seed)

if(torch.cuda.is_available()):
    print("GPU: ",torch.cuda.get_device_name(1), " is available, Switching now.")
else:
    print("GPU is not available, using CPU.")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device is now: ", device)

if temp < 1e-3:
    exit(-999)

with open(CHECKPOINT_PREFIX, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

#corpus = data.Corpus(args.data)
#ntokens = len(corpus.dictionary)
songs = []
with open(os.path.join(cwd, 'dataset', 'irish.abc'), 'r') as f:
    text = f.read()
    songs = extract_song_snippet(text)

# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs)

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")

ntokens = len(vocab)

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


is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    exit(-999)
    hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(GENERATION_PREFIX, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(gen_length):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(temp).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(temp).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            word = idx2char[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ''))

            if i % log_interval == 0:
                print('| Generated {}/{} words'.format(i, gen_length))
