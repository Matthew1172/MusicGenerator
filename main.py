from torch import nn

from MusicData import *
from common import DATASET_PREFIX_PRETTY
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from Transformer_Model import *

pin_memory=False
num_workers=0

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
epochs = 20
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

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def main(rank, world_size):
    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader
    dataset = ABCMusicDataset(DATASET_PREFIX_PRETTY)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)

    ntokens = len(dataset.dic.idx2word)


    # instantiate the model(it's your own model) and move it to the right device
    model = TransformerModel(ntokens, emsize, num_heads, hidden_units, nlayers, dropout).to(rank)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    #################### The above is defined previously

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        #dataloader.sampler.set_epoch(epoch)
        sampler.set_epoch(epoch)

        for step, x in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            label = x['label']

            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
    cleanup()

if __name__ == '__main__':

    # suppose we have 3 gpus
    world_size = 2
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )

'''
dataset = ABCMusicDataset(DATASET_PREFIX_PRETTY)

train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                num_replicas=2,
                                                                rank=1)
train_loader = torch.utils.data.DataLoader(dataset=train_sampler,
                                           batch_size=32,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=True,
                                           sampler=train_sampler)
train_loader = get_valid_loader(dataset, 32)
target = next(iter(train_loader))
'''