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

    with open(path, 'wb') as f:
        img.save(f)

DEBUG_FAST_NUMBER = 1000

class DiffDataset:
    def __init__(self, ds_name='fixmypose', split='train'):
        self.ds_name = ds_name
        self.split = split
        self.data = json.load(
            open(os.path.join(DATA_ROOT, self.ds_name, self.split + "_ret_NC.json"))
        )

        self.tok = Tokenizer()
        self.tok.load(os.path.join(DATA_ROOT, self.ds_name, "vocab_ret_hindi_NC.txt"))



class TorchDataset(Dataset):
    def __init__(self, dataset, max_length=80, 
                 img0_transform=None, img1_transform=None):
        self.dataset = dataset
        self.name = dataset.ds_name + "_" + dataset.split
        self.tok = dataset.tok
        self.max_length = max_length
        self.img0_trans, self.img1_trans = img0_transform, img1_transform

        f = h5py.File(os.path.join(DATA_ROOT, self.dataset.ds_name, 
            self.dataset.split + "_ret_NC_pixels.hdf5"), 'r')
        if args.fast:
            self.img0_pixels = f['img0'][:DEBUG_FAST_NUMBER]
            self.img1_pixels = f['img1'][:DEBUG_FAST_NUMBER]
        else:
            self.img0_pixels = f['img0']
            self.trg0_pixels = f['trg0']
            self.trg1_pixels = f['trg1']
            self.trg2_pixels = f['trg2']
            self.trg3_pixels = f['trg3']
            self.trg4_pixels = f['trg4']
            self.trg5_pixels = f['trg5']
            self.trg6_pixels = f['trg6']
            self.trg7_pixels = f['trg7']
            self.trg8_pixels = f['trg8']
            self.trg9_pixels = f['trg9']
            assert len(self.img0_pixels) == len(self.dataset.data), "%d, %d" % (len(self.img0_pixels),
                                                                                len(self.dataset.data))
            assert len(self.trg0_pixels) == len(self.dataset.data)
        


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
        ans_id = datum['ans_idx']
        
        # Load Image
        img_id = self.id2imgid[item]   

        img0 = torch.from_numpy(self.img0_pixels[img_id])      
        trg0 = torch.from_numpy(self.trg0_pixels[img_id])
        trg1 = torch.from_numpy(self.trg1_pixels[img_id])
        trg2 = torch.from_numpy(self.trg2_pixels[img_id])
        trg3 = torch.from_numpy(self.trg3_pixels[img_id])
        trg4 = torch.from_numpy(self.trg4_pixels[img_id])
        trg5 = torch.from_numpy(self.trg5_pixels[img_id])
        trg6 = torch.from_numpy(self.trg6_pixels[img_id])
        trg7 = torch.from_numpy(self.trg7_pixels[img_id])
        trg8 = torch.from_numpy(self.trg8_pixels[img_id])
        trg9 = torch.from_numpy(self.trg9_pixels[img_id])

        img0ID = datum['img0'].split("/")[-1].split(".")[0]
        trg0ID = datum['trg0'].split("/")[-1].split(".")[0]
        trg1ID = datum['trg1'].split("/")[-1].split(".")[0]
        trg2ID = datum['trg2'].split("/")[-1].split(".")[0]
        trg3ID = datum['trg3'].split("/")[-1].split(".")[0]
        trg4ID = datum['trg4'].split("/")[-1].split(".")[0]
        trg5ID = datum['trg5'].split("/")[-1].split(".")[0]
        trg6ID = datum['trg6'].split("/")[-1].split(".")[0]
        trg7ID = datum['trg7'].split("/")[-1].split(".")[0]
        trg8ID = datum['trg8'].split("/")[-1].split(".")[0]
        trg9ID = datum['trg9'].split("/")[-1].split(".")[0]


        if False or self.dataset.split == "train":
            sent = datum['sent_hindi']
        else:
            sent = datum['sents_hindi'][0]
            
        inst = self.tok.encode(sent)
        length = len(inst)
        a = np.ones((self.max_length), np.int64) * self.tok.pad_id

        if length < self.max_length:        
            a[: length] = inst

            length = length
        else:                                        
            a[:] = inst[:self.max_length]

            length = self.max_length

        # Lang: numpy --> torch
        inst = torch.from_numpy(a)
        leng = torch.tensor(length)
        trg = (trg0,trg1,trg2,trg3,trg4,trg5,trg6,trg7,trg8,trg9)

        return uid, img0, trg, inst, leng, ans_id
        