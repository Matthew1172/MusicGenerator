import torch.distributions.distribution
from torch.optim.lr_scheduler import ExponentialLR
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from Transformer_Model import *
import data
import os
import time
from common import DATASETS, DATASET, CHECKPOINT_DIR, CHECKPOINT_PREFIX

#size of word embeddings
emsize = 512
#number of hidden units per layer
hidden_units = 2048
#number of layers
nlayers = 2
#initial learning rate
learning_rate = 1e-1
#momentum for SGD
momentum = 0.9
#upper epoch limit
epochs = 200
#batch size
batch_size = 10
#sequence length
bptt = 150
#dropout applied to layers (0 = no dropout)
dropout = 0.65
#report interval
log_interval = 200
#the number of heads in the encoder/decoder of the transformer model
num_heads = 8
eval_batch_size = 10

assert os.path.exists(DATASETS)
assert os.path.exists(DATASET)
assert os.path.exists(CHECKPOINT_DIR)

myCorpus = data.Corpus(DATASET, from_bin=bin)
print("Found {} bad songs out of {}.".format(myCorpus.bad, myCorpus.total))

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

train_data = batchify(myCorpus.train, batch_size)
val_data = batchify(myCorpus.valid, eval_batch_size)
test_data = batchify(myCorpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(myCorpus.dictionary)
loss_fn = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def demo_checkpoint(rank, world_size):
    # Loop over epochs.
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            # create default process group
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
            # create local model
            model = TransformerModel(ntokens, emsize, num_heads, hidden_units, nlayers, dropout).to(
                rank)

            # construct DDP model
            ddp_model = DDP(model, device_ids=[rank])
            # define loss function and optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
            scheduler = ExponentialLR(optimizer, gamma=0.9)

            # forward pass
            total_loss = 0.
            start_time = time.time()
            for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
                data, targets = get_batch(train_data, i)
                optimizer.zero_grad()
                output = ddp_model(data.to(rank))
                output = output.view(-1, ntokens)
                # backward pass
                loss = loss_fn(output, targets).backward()
                # update parameters
                optimizer.step()

                total_loss += loss.item()

                if batch % log_interval == 0 and batch > 0:
                    pass
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f} | meanloss: {:5.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr(),
                                  elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss),
                    loss.cpu().detach().numpy().mean()))
                total_loss = 0
                start_time = time.time()

            scheduler.step()
            model.eval()
            total_loss = 0.
            with torch.no_grad():
                for i in range(0, val_data.size(0) - 1, bptt):
                    data, targets = get_batch(val_data, i)
                    output = model(data)
                    output = output.view(-1, ntokens)
                    total_loss += len(data) * loss_fn(output, targets).item()
            val_loss = total_loss / (len(val_data) - 1)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if rank == 0:
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(CHECKPOINT_PREFIX, 'wb') as f:
                        torch.save(ddp_model.state_dict(), f)
                    best_val_loss = val_loss
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def main():
    world_size = 2
    mp.spawn(demo_checkpoint,
        args=(world_size,),
        nprocs=world_size,
        join=True)


if __name__ == "__main__": main()