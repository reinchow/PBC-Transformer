import os

import nltk
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
np.random.seed(1234)


class DataLoader(data.Dataset):
    def __init__(self, root, json, vocab, transform=None):
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        captions1 = coco.anns[ann_id]['caption']
        rdn_index = np.random.choice(len(captions1), 1)[0]
        captions = captions1[rdn_index]
        path = coco.anns[ann_id]['image_id']
        labels = int(coco.anns[ann_id]['labels'])

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(captions).lower())
        caption1 = []
        caption1.append(vocab('<start>'))
        caption1.extend([vocab(token) for token in tokens])
        caption1.append(vocab('<end>'))
        target = torch.Tensor(caption1)

        return image, target, labels

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, labels = zip(*data)

    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    labels = torch.Tensor(labels).long()
    return images, targets, lengths, labels


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    coco = DataLoader(root=root, json=json, vocab=vocab, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
