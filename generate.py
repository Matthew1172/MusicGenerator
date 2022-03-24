import torch
import data
import os
from tqdm import tqdm
import regex as re

seed = 0
temp = 1.0
gen_length = 200
log_interval = 200

# Checkpoint location:
CHECKPOINT_DIR = 'training_checkpoints_pytorch'
CHECKPOINT_PREFIX = 'my_ckpt.pth'

cwd = os.getcwd()
CHECKPOINT_DIR = os.path.join(cwd, CHECKPOINT_DIR)
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)
GENERATION_PREFIX = os.path.join(cwd, "generated.txt")

dataset = "./dataset/irish"

# Set the random seed manually for reproducibility.
torch.manual_seed(seed)
if(torch.cuda.is_available()):
    print("GPU: ",torch.cuda.get_device_name(0), " is available, Switching now.")
else:
    print("GPU is not available, using CPU.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is now: ", device)

#-999 : temperature has to be greater or equal 1e-3.
if temp < 1e-3:
    exit(-999)

with open(CHECKPOINT_PREFIX, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

corpus = data.Corpus(dataset)
ntokens = len(corpus.dictionary)

input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

'''TODO: create music 21 score'''

with open(GENERATION_PREFIX, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in tqdm(range(gen_length)):
            output = model(input, False)
            word_weights = output[-1].squeeze().div(temp).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            input = torch.cat([input, word_tensor], 0)

            word = corpus.dictionary.idx2word[word_idx]

            '''TODO: construct a time signature, clef, and key signature object and append it to the score'''
            '''word is either a clef, key signature, time signature, note, or barline'''

            #outf.write(word + ('\n' if i % 20 == 19 else ''))
            outf.write(word)

            if i % log_interval == 0:
                print('| Generated {}/{} words'.format(i, gen_length))