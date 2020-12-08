from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
from param import args
from tok import Tokenizer
from utils import BufferLoader
import copy
from PIL import Image
import json
import random
import os
import numpy as np
import torch

DATA_ROOT = "dataset/"

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_saver(img, path):
    # print(img, path)
    with open(path, 'wb') as f:
        img.save(f)

DEBUG_FAST_NUMBER = 1000

class DiffDataset:
    def __init__(self, ds_name='fixmypose', split='train'):
        self.ds_name = ds_name
        self.split = split
        self.data = json.load(
            open(os.path.join(DATA_ROOT, self.ds_name, self.split + "_NC.json"))
        )

        self.tok = Tokenizer()
        self.tok.load(os.path.join(DATA_ROOT, self.ds_name, "vocab_hindi_NC.txt"))


class TorchDataset(Dataset):
    def __init__(self, dataset, max_length=80, 
                 img0_transform=None, img1_transform=None):
        self.dataset = dataset
        self.name = dataset.ds_name + "_" + dataset.split
        self.tok = dataset.tok
        self.max_length = max_length
        self.img0_trans, self.img1_trans = img0_transform, img1_transform


        f = h5py.File(os.path.join(DATA_ROOT, self.dataset.ds_name, 
            self.dataset.split + "_NC_pixels.hdf5"), 'r')
        if args.fast:
            self.img0_pixels = f['img0'][:DEBUG_FAST_NUMBER]
            self.img1_pixels = f['img1'][:DEBUG_FAST_NUMBER]
        else:
            self.img0_pixels = f['img0']
            self.img1_pixels = f['img1']
            assert len(self.img0_pixels) == len(self.dataset.data), "%d, %d" % (len(self.img0_pixels),
                                                                                len(self.dataset.data))
            assert len(self.img1_pixels) == len(self.dataset.data)
        
        if False or self.dataset.split == "train":
            self.train_data = []
            self.id2imgid = {}
            for i, datum in enumerate(self.dataset.data):
                if args.fast and i >= DEBUG_FAST_NUMBER:     
                    break
                for sent in datum['sents_hindi']:
                    new_datum = datum.copy()
                    new_datum.pop('sents_hindi')
                    new_datum['sent_hindi'] = sent
                    self.id2imgid[len(self.train_data)] = i     
                    self.train_data.append(new_datum)

        else:
            self.train_data = []
            self.id2imgid = {}
            for i, datum in enumerate(self.dataset.data):
                if args.fast and i >= DEBUG_FAST_NUMBER:     
                    break

                self.id2imgid[len(self.train_data)] = i    
                self.train_data.append(datum)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        datum = self.train_data[item]
        uid = datum['uid']
        
        # Load Image
        img_id = self.id2imgid[item]        

        img0 = torch.from_numpy(self.img0_pixels[img_id])      
        img1 = torch.from_numpy(self.img1_pixels[img_id])
        
        img0ID = datum['img0'].split("/")[-1].split(".")[0]
        img1ID = datum['img1'].split("/")[-1].split(".")[0]

        if False or self.dataset.split == "train":
            sent = datum['sent_hindi']
        else:
            sent = datum['sents_hindi'][0]


        sent = sent.replace(".", "").replace(",", "") 
        inst = self.tok.encode(sent)
        length = len(inst)
        a = np.ones((self.max_length), np.int64) * self.tok.pad_id
        a[0] = self.tok.bos_id
        if length + 2 < self.max_length:        
            a[1: length+1] = inst
            a[length+1] = self.tok.eos_id
            length = 2 + length
        else:                                           
            a[1: -1] = inst[:self.max_length-2]
            a[self.max_length-1] = self.tok.eos_id      
            length = self.max_length

        inst = torch.from_numpy(a)
        leng = torch.tensor(length)

        return uid, img0, img1, inst, leng
        
