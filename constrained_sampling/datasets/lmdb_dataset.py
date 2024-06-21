import torch.utils.data as data
import numpy as np
import lmdb
import os
import io
from PIL import Image


def num_samples(dataset, train):
    if dataset == 'celeba':
        return 27000 if train else 3000
    elif dataset == 'celeba64':
        return 162770 if train else 19867
    elif dataset == 'imagenet-oord':
        return 1281147 if train else 50000
    elif dataset == 'ffhq':
        return 63000 if train else 7000
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)


class LMDBDataset(data.Dataset):
    def __init__(self, root, name='', split='train', transform=None, is_encoded=False):
        self.name = name
        self.transform = transform
        self.split = split
        if self.split == 'train':
            lmdb_path = os.path.join(root, 'train.lmdb')
        elif self.split == 'val':
            lmdb_path = os.path.join(root, 'validation.lmdb')
        else:
            lmdb_path = os.path.join(f'{root}.lmdb')

        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

    def __getitem__(self, index):
        target = 0
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            # Dataset SNIPS
            key = f'{256}-{str(index).zfill(5)}'.encode('utf-8')
            data = txn.get(key)

            # data = txn.get(str(index).encode())
            # if self.is_encoded:
            img = Image.open(io.BytesIO(data))
            img = img.convert('RGB')
            # else:
            #     img = np.asarray(data, dtype=np.uint8)
            #     # assume data is RGB
            #     size = int(np.sqrt(len(img) / 3))
            #     img = np.reshape(img, (size, size, 3))
            #     img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, {'index': str(index).zfill(5) + '.png'}

    def __len__(self):
        if hasattr(self, 'length'):
            return self.length
        else:
            with self.data_lmdb.begin() as txn:
                self.length = txn.stat()['entries']
            return self.length


# class LMDBDataset(data.Dataset):
#     def __init__(self, root, name='', split='train', transform=None, is_encoded=False):
#         path = os.path.join(root, 'validation.lmdb')
#         self.env = lmdb.open(
#             path,
#             max_readers=32,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )

#         if not self.env:
#             raise IOError('Cannot open lmdb dataset', path)

#         with self.env.begin(write=False) as txn:
#             self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

#         self.resolution = 256
#         self.transform = transform
#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         with self.env.begin(write=False) as txn:
#             key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
#             img_bytes = txn.get(key)

#         buffer = io.BytesIO(img_bytes)
#         img = Image.open(buffer)
#         img = img.convert('RGB')
#         target = 0

#         return img, target, index