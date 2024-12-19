# encoding: utf-8

import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid, *other) or (img_path, mask_path, pid, camid, *other).
    - ds_type (str): dataset type; determines data_source entry format.
    - batch_size (int): number of examples in a batch.
    - num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, ds_type, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, e in enumerate(self.data_source):
            if ds_type == "object-masked":
                _, _, pid, _, *other = e
            elif ds_type in ["person", "object"]:
                _, pid, _, *other = e
            else:
                raise NotImplementedError
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomClassSampler(Sampler):
    """
    Randomly sample C classes, the for each class,
    randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is C*N*K.
    Each batch must have its identities evenly split between C object classes.

    Args:
    - data_source (list): list of (img_path, pid, camid, *other) or (img_path, mask_path, pid, camid, *other).
    - ds_type (str): dataset type; determines data_source entry format.
    - batch_size (int): number of examples in a batch.
    - num_instances (int): number of instances per identity in a batch.
    - num_classes (int): number of object classes in a batch.
    """

    def __init__(self, data_source, ds_type, batch_size, num_instances, num_classes):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.num_classes = num_classes
        self.num_pids_per_class = self.num_pids_per_batch // self.num_classes
        assert self.num_pids_per_batch % self.num_classes == 0
        # create maps of class -> IDs and ID -> Images
        self.index_dic = defaultdict(list)
        self.class_dict = defaultdict(set)
        for index, e in enumerate(self.data_source):
            if ds_type == "object-masked":
                _, _, pid, _, *other = e
            elif ds_type in ["person", "object"]:
                _, pid, _, *other = e
            else:
                raise NotImplementedError
            class_ = other[0]
            self.class_dict[class_].add(pid)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.classes = list(self.class_dict)

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        # create image batches of size num_instances
        # discard remainder images, if not enough remain to fill a batch
        # use duplicate images, if not even one batch can be created
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        class_dict = copy.deepcopy(self.class_dict)
        # sampling weights for classes proportional to number of instanes in class
        class_weights = np.array([len(class_dict[class_]) for class_ in self.classes])
        class_weights[class_weights < self.num_pids_per_class] = 0
        final_idxs = []

        # sample classes and IDs, collect image batches per ID
        while len(class_weights[class_weights > 0]) >= self.num_classes:
            class_choices = np.random.choice(len(self.classes), self.num_classes, replace=False, p=class_weights/np.sum(class_weights))
            for idx in class_choices:
                class_ = self.classes[idx]
                selected_pids = random.sample(class_dict[class_], self.num_pids_per_class)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        class_dict[class_].remove(pid)
                        class_weights[idx] -= 1
                        if len(class_dict[class_]) < self.num_pids_per_class:
                            class_weights[idx] = 0

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class ClassBalancedSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Ensure, that identities are sampled equally from all classes,
    using duplicates if necessary.

    Args:
    - data_source (list): list of (img_path, pid, camid, *other) or (img_path, mask_path, pid, camid, *other).
    - ds_type (str): dataset type; determines data_source entry format.
    - batch_size (int): number of examples in a batch.
    - num_instances (int): number of instances per identity in a batch.
    """
    def __init__(self, data_source, ds_type, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        # create maps of class -> IDs and ID -> Images
        self.index_dic = defaultdict(list)
        self.class_dict = defaultdict(set)
        for index, e in enumerate(self.data_source):
            if ds_type == "object-masked":
                _, _, pid, _, *other = e
            elif ds_type in ["person", "object"]:
                _, pid, _, *other = e
            else:
                raise NotImplementedError
            class_ = other[0]
            self.class_dict[class_].add(pid)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.classes = list(self.class_dict)

        # estimate number of examples in an epoch
        self.length = 0
        max_num = max(len(x) for x in self.class_dict.values())
        for class_ in self.classes:
            if len(self.class_dict[class_]) == max_num:
                for pid in self.class_dict[class_]:
                    idxs = self.index_dic[pid]
                    num = len(idxs)
                    if num < self.num_instances:
                        num = self.num_instances
                    self.length += num - num % self.num_instances
        self.length *= len(self.classes)

    def __iter__(self):
        # create image batches of size num_instances
        # discard remainder images, if not enough remain to fill a batch
        # use duplicate images, if not even one batch can be created
        batch_class_idxs_dict = {}
        batch_num_dict = defaultdict(int)
        for class_ in self.classes:
            batch_class_idxs_dict[class_] = defaultdict(list)
            for pid in self.class_dict[class_]:
                idxs = copy.deepcopy(self.index_dic[pid])
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_class_idxs_dict[class_][pid].append(batch_idxs)
                        batch_num_dict[pid] += 1
                        batch_idxs = []

        # create duplicates of IDs untill all classes have same number of IDs
        #  to achieve balanced sampling
        batch_idxs_dict = {}
        max_num = max([len(x) for x in batch_class_idxs_dict.values()])
        for class_ in self.classes:
            pid_dict = batch_class_idxs_dict[class_]
            if len(pid_dict) < max_num:
                add_pids = np.random.choice(list(pid_dict.keys()), size=max_num-len(pid_dict), replace=True)
                for pid in add_pids:
                    idxs = copy.deepcopy(pid_dict[pid][:batch_num_dict[pid]])
                    idxs = [id_ for sample in idxs for id_ in sample]
                    random.shuffle(idxs)
                    batch_idxs = []
                    for idx in idxs:
                        batch_idxs.append(idx)
                        if len(batch_idxs) == self.num_instances:
                            pid_dict[pid].append(batch_idxs)
                            batch_idxs = []
            batch_idxs_dict.update(pid_dict)

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        # sample IDs, collect image batches per ID
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
