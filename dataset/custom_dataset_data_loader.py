import torch.utils.data
from dataset.base_data_loader import BaseDataLoader
from utils.utils import collate_fn
import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist
import numpy as np
import random


def worker_init_fn(worker_id):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = 4 * dist.get_rank() + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class CustomDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    For each batch, sample mixtures containing 2,3,4,5 speakers.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_batches = math.ceil(len(self.dataset) / self.num_replicas / 16)
        self.total_size = self.num_batches * self.num_replicas * 16
        self.shuffle = shuffle
        self.seed = seed

    def get_indices(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_batches * 16

        indices_ = []
        for i in range(self.num_batches):
            indices_.append([(indices[16 * i], indices[16 * i + 1]),  # 2 speakers
                             (indices[16 * i + 2], indices[16 * i + 3]),  # 2 speakers
                             (indices[16 * i + 4], indices[16 * i + 5], indices[16 * i + 6]),  # 3 speakers
                             (indices[16 * i + 7], indices[16 * i + 8], indices[16 * i + 9], indices[16 * i + 10]),  # 4 speakers
                             (indices[16 * i + 11], indices[16 * i + 12], indices[16 * i + 13], indices[16 * i + 14], indices[16 * i + 15])])  # 5 speakers

        return indices_

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler and generate mixture list at the beginning of epoch.
        """
        self.epoch = epoch
        self.indices = self.get_indices()

    def __iter__(self):
        return iter(self.indices)


def CreateDataset(opt):
    if opt.model == 'audioVisual':
        from dataset.audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.model)

    if opt.rank == 0:
        print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        sampler = CustomDistributedSampler(self.dataset, shuffle=True)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            num_workers=int(opt.nThreads),
            collate_fn=collate_fn,
            sampler=sampler,
            worker_init_fn=worker_init_fn)

    def set_epoch(self, epoch):
        self.dataloader.sampler.set_epoch(epoch)

    def dataset(self):
        return self.dataset

    def _get_indices(self):
        return self.dataloader.sampler.indices

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
