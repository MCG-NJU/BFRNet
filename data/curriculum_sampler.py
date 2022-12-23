#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data
import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class CurriculumDistributedSampler(Sampler):
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

    def __init__(self, dataset, curriculum_sample, num_replicas=None, rank=None, shuffle=True, seed=0):
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
        self.curriculum_sample = curriculum_sample
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # 2mix: [2] * 8,  3mix: [2] * 5 + [3] * 2,  4mix: [2] * 3 + [3] * 2 + [4] * 1,  5mix: [2] * 2 + [3] * 1 + [4] * 1 + [5] * 1
        # 以16个视频为一个batch, 一个batch里有多种mix组合
        self.num_batches = math.ceil(len(self.dataset) / self.num_replicas / 16)  # 每个gpu上的batch数量,一个batch包括5个samples,16个视频
        self.total_size = self.num_batches * self.num_replicas * 16  # 一个epoch所有gpu见到的视频总数
        # self.num_samples = self.num_batches * 5  # 每个gpu上的samples
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_batches

    def _2mix(self, indices):
        indices_ = []
        for i in range(self.num_batches):
            indices_.append([[indices[16 * i + 0], indices[16 * i + 1]],
                             [indices[16 * i + 2], indices[16 * i + 3]],
                             [indices[16 * i + 4], indices[16 * i + 5]],
                             [indices[16 * i + 6], indices[16 * i + 7]],
                             [indices[16 * i + 8], indices[16 * i + 9]],
                             [indices[16 * i +10], indices[16 * i +11]],
                             [indices[16 * i +12], indices[16 * i +13]],
                             [indices[16 * i +14], indices[16 * i +15]]])
        return indices_

    def _2mix_3mix(self, indices):
        indices_ = []
        for i in range(self.num_batches):
            indices_.append([[indices[16 * i + 0], indices[16 * i + 1]],
                             [indices[16 * i + 2], indices[16 * i + 3]],
                             [indices[16 * i + 4], indices[16 * i + 5]],
                             [indices[16 * i + 6], indices[16 * i + 7]],
                             [indices[16 * i + 8], indices[16 * i + 9]],
                             [indices[16 * i +10], indices[16 * i + 11], indices[16 * i + 12]],
                             [indices[16 * i +13], indices[16 * i + 14], indices[16 * i + 15]]])
        return indices_

    def _2mix_3mix_4mix(self, indices):
        indices_ = []
        for i in range(self.num_batches):
            indices_.append([[indices[16 * i + 0], indices[16 * i + 1]],
                             [indices[16 * i + 2], indices[16 * i + 3]],
                             [indices[16 * i + 4], indices[16 * i + 5]],
                             [indices[16 * i + 6], indices[16 * i + 7], indices[16 * i + 8]],
                             [indices[16 * i + 9], indices[16 * i +10], indices[16 * i +11]],
                             [indices[16 * i +12], indices[16 * i +13], indices[16 * i +14], indices[16 * i +15]]])
        return indices_

    def _2mix_3mix_4mix_5mix(self, indices):
        indices_ = []
        for i in range(self.num_batches):
            indices_.append([[indices[16 * i + 0], indices[16 * i + 1]],
                             [indices[16 * i + 2], indices[16 * i + 3]],
                             [indices[16 * i + 4], indices[16 * i + 5], indices[16 * i + 6]],
                             [indices[16 * i + 7], indices[16 * i + 8], indices[16 * i + 9], indices[16 * i +10]],
                             [indices[16 * i +11], indices[16 * i +12], indices[16 * i +13], indices[16 * i +14], indices[16 * i + 15]]])
        return indices_

    def set_epoch(self, epoch):
        # set epoch and generate iter
        self.epoch = epoch

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

        if epoch < self.curriculum_sample[0]:  # sample all 2mix
            self.indices = self._2mix(indices)
        elif epoch < self.curriculum_sample[1]:  # sample 2mix and 3mix
            self.indices = self._2mix_3mix(indices)
        elif epoch < self.curriculum_sample[2]:  # sample 2mix, 3mix and 4mix
            self.indices = self._2mix_3mix_4mix(indices)
        else:  # sample 2mix, 3mix, 4mix, and 5mix
            self.indices = self._2mix_3mix_4mix_5mix(indices)


class CurriculumDistributedSampler2(Sampler):
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

    def __init__(self, dataset, curriculum_sample, num_replicas=None, rank=None, shuffle=True, seed=0):
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
        self.curriculum_sample = curriculum_sample
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # 2mix: [2]*8,  3mix: [2]*5 + [3]*3,  4mix: [2]*3 + [3]*3 + [4]*2,  5mix: [2]*2 + [3]*2 + [4]*2 + [5]*2
        # 16, 19, 23, 28
        # 一个batch里有多种mix组合, 不同的batch含有的总人数不一定一样
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_batches

    def _get_indices(self):
        return self.indices

    def _2mix(self, indices):
        # [2] * 8
        indices_ = []
        for i in range(self.num_batches):
            indices_.append([[indices[16 * i + 0], indices[16 * i + 1]],
                             [indices[16 * i + 2], indices[16 * i + 3]],
                             [indices[16 * i + 4], indices[16 * i + 5]],
                             [indices[16 * i + 6], indices[16 * i + 7]],
                             [indices[16 * i + 8], indices[16 * i + 9]],
                             [indices[16 * i +10], indices[16 * i +11]],
                             [indices[16 * i +12], indices[16 * i +13]],
                             [indices[16 * i +14], indices[16 * i +15]]])
        return indices_

    def _2mix_3mix(self, indices):
        # [2] * 5 + [3] * 3
        indices_ = []
        for i in range(self.num_batches):
            indices_.append([[indices[19 * i + 0], indices[19 * i + 1]],
                             [indices[19 * i + 2], indices[19 * i + 3]],
                             [indices[19 * i + 4], indices[19 * i + 5]],
                             [indices[19 * i + 6], indices[19 * i + 7]],
                             [indices[19 * i + 8], indices[19 * i + 9]],
                             [indices[19 * i +10], indices[19 * i + 11], indices[19 * i + 12]],
                             [indices[19 * i +13], indices[19 * i + 14], indices[19 * i + 15]],
                             [indices[19 * i +16], indices[19 * i + 17], indices[19 * i + 18]]])
        return indices_

    def _2mix_3mix_4mix(self, indices):
        # [2] * 3 + [3] * 3 + [4] * 2
        indices_ = []
        for i in range(self.num_batches):
            indices_.append([[indices[23 * i + 0], indices[23 * i + 1]],
                             [indices[23 * i + 2], indices[23 * i + 3]],
                             [indices[23 * i + 4], indices[23 * i + 5]],
                             [indices[23 * i + 6], indices[23 * i + 7], indices[23 * i + 8]],
                             [indices[23 * i + 9], indices[23 * i +10], indices[23 * i +11]],
                             [indices[23 * i +12], indices[23 * i +13], indices[23 * i +14]],
                             [indices[23 * i +15], indices[23 * i +16], indices[23 * i +17], indices[23 * i +18]],
                             [indices[23 * i +19], indices[23 * i +20], indices[23 * i +21], indices[23 * i +22]]])
        return indices_

    def _2mix_3mix_4mix_5mix(self, indices):
        # [2] * 2 + [3] * 2 + [4] * 2 + [5] * 2
        indices_ = []
        for i in range(self.num_batches):
            indices_.append([[indices[28 * i + 0], indices[28 * i + 1]],
                             [indices[28 * i + 2], indices[28 * i + 3]],
                             [indices[28 * i + 4], indices[28 * i + 5], indices[28 * i + 6]],
                             [indices[28 * i + 7], indices[28 * i + 8], indices[28 * i + 9]],
                             [indices[28 * i +10], indices[28 * i +11], indices[28 * i +12], indices[28 * i +13]],
                             [indices[28 * i +14], indices[28 * i +15], indices[28 * i +16], indices[28 * i +17]],
                             [indices[28 * i +18], indices[28 * i +19], indices[28 * i +20], indices[28 * i +21], indices[28 * i +22]],
                             [indices[28 * i +23], indices[28 * i +24], indices[28 * i +25], indices[28 * i +26], indices[28 * i +27]]])
        return indices_

    def set_epoch(self, epoch):
        # print(f"rank:{dist.get_rank()}, set_epoch: {epoch}", flush=True)
        # set epoch and generate iter
        self.epoch = epoch

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if epoch < self.curriculum_sample[0]:
            person_num_per_batch = 16
        elif epoch < self.curriculum_sample[1]:
            person_num_per_batch = 19
        elif epoch < self.curriculum_sample[2]:
            person_num_per_batch = 23
        else:
            person_num_per_batch = 28

        self.num_batches = math.ceil(len(self.dataset) / self.num_replicas / person_num_per_batch)  # 每个gpu经历的batch数量
        self.total_size = self.num_batches * self.num_replicas * person_num_per_batch

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_batches * person_num_per_batch

        if epoch < self.curriculum_sample[0]:  # sample all 2mix
            self.indices = self._2mix(indices)
        elif epoch < self.curriculum_sample[1]:  # sample 2mix and 3mix
            self.indices = self._2mix_3mix(indices)
        elif epoch < self.curriculum_sample[2]:  # sample 2mix, 3mix and 4mix
            self.indices = self._2mix_3mix_4mix(indices)
        else:  # sample 2mix, 3mix, 4mix, and 5mix
            self.indices = self._2mix_3mix_4mix_5mix(indices)

        # print(f"rank:{dist.get_rank()} finish set_epoch{epoch}", flush=True)
