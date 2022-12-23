import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.curriculum_sampler import *
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

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
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
        # #2:#3:#4:#5 = 2:1:1:1, 每两个#2之后跟1个#3,#4,#5, 每批次16个视频
        self.num_batches = math.ceil(len(self.dataset) / self.num_replicas / 16)  # 每个gpu上的batch数量,一个batch包括5个samples,16个视频
        self.total_size = self.num_batches * self.num_replicas * 16  # 一个epoch所有gpu见到的视频总数
        # self.num_samples = self.num_batches * 5  # 每个gpu上的samples
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
            # indices_ += [(indices[16*i], indices[16*i+1]), (indices[16*i+2], indices[16*i+3])]
            # indices_ += [(indices[16*i+4], indices[16*i+5], indices[16*i+6])]
            # indices_ += [(indices[16*i+7], indices[16*i+8], indices[16*i+9], indices[16*i+10])]
            # indices_ += [(indices[16*i+11], indices[16*i+12], indices[16*i+13], indices[16*i+14], indices[16*i+15])]
            indices_.append([(indices[16 * i], indices[16 * i + 1]),
                             (indices[16 * i + 2], indices[16 * i + 3]),
                             (indices[16 * i + 4], indices[16 * i + 5], indices[16 * i + 6]),
                             (indices[16 * i + 7], indices[16 * i + 8], indices[16 * i + 9], indices[16 * i + 10]),
                             (indices[16 * i + 11], indices[16 * i + 12], indices[16 * i + 13], indices[16 * i + 14], indices[16 * i + 15])])
        # indices = indices_

        return indices_

    def __len__(self):
        # return self.num_samples
        return self.num_batches

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self.indices = self.get_indices()

    def __iter__(self):
        return iter(self.indices)


def CreateDataset(opt):
    if opt.model == 'audioVisual':
        from data.audioVisual_dataset import AudioVisualDataset
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
        # if opt.mode == "train":
        if opt.sampler_type == "normal":
            sampler = CustomDistributedSampler(self.dataset, shuffle=True)
        elif opt.sampler_type == "curriculum":
            sampler = CurriculumDistributedSampler(self.dataset, opt.curriculum_sample, shuffle=True)
        elif opt.sampler_type == "curriculum2":
            sampler = CurriculumDistributedSampler2(self.dataset, opt.curriculum_sample, shuffle=True)
        else:
            assert ValueError(f'wrong opt.sampler_type: {opt.sampler_type}')
        if opt.mode == 'train':
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                num_workers=int(opt.nThreads),
                collate_fn=collate_fn,
                sampler=sampler,
                worker_init_fn=worker_init_fn)
        # elif opt.mode == 'val':
        # if opt.sampler_type == "normal":
        #     sampler = CustomDistributedSampler(self.dataset, shuffle=True)
        # elif opt.sampler_type == "curriculum":
        #     sampler = CurriculumDistributedSampler(self.dataset, [0, 0, 0], shuffle=True)
        # else:
        #     assert ValueError(f'wrong opt.sampler_type: {opt.sampler_type}')
        elif opt.mode == 'val':
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
