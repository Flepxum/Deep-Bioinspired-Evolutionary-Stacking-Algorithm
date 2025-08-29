# -*- coding: utf-8 -*-

import random

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms


def get_data_loader(data_dir,
                    batch_size,
                    random_seed,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True):
    # define transforms
    trans = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [255, 255, 255]),
    ])

    if shuffle:
        np.random.seed(random_seed)

    # load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=trans)

    meta_train_size = round(len(dataset) * 0.75)
    meta_valid_size = len(dataset) - meta_train_size
    meta_set = torch.utils.data.random_split(dataset, [meta_train_size, meta_valid_size],
                                             torch.Generator().manual_seed(random.randint(0, 100)))

    basic_set = meta_set[0]
    meta_valid_set = meta_set[1]

    l = round(len(basic_set) / 5)

    set = torch.utils.data.random_split(basic_set, [l, l, l, l, len(basic_set) - 4 * l],
                                        torch.Generator().manual_seed(random.randint(0, 100)))

    basic_train_set = []
    basic_valid_set = []
    basic_train_set.append(set[0] + set[1] + set[2] + set[3])
    basic_train_set.append(set[0] + set[1] + set[2] + set[4])
    basic_train_set.append(set[0] + set[1] + set[3] + set[4])
    basic_train_set.append(set[0] + set[2] + set[3] + set[4])
    basic_train_set.append(set[1] + set[2] + set[3] + set[4])
    basic_valid_set.append(set[4])
    basic_valid_set.append(set[3])
    basic_valid_set.append(set[2])
    basic_valid_set.append(set[1])
    basic_valid_set.append(set[0])

    train_loader = []
    valid_loader = []

    for i in range(5):
        train_loader.append(torch.utils.data.DataLoader(
            basic_train_set[i], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
        )
        )
        # the batch_size for testing is set as the valid_size
        valid_loader.append(torch.utils.data.DataLoader(
            basic_valid_set[i], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
        )
        )

    meta_valid_loader = torch.utils.data.DataLoader(
        meta_valid_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, meta_valid_loader
