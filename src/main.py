import random
import numpy as np

import torch
import torchvision.transforms as transforms

from param import args
from speaker import Speaker

from data import DiffDataset, TorchDataset

# Set the seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Image Transformation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img_transform = transforms.Compose([
    transforms.Resize((args.resize, args.resize)),
    transforms.ToTensor(),
    normalize
])

args.workers = 1    

# Loading Dataset
def get_tuple(ds_name, split, shuffle=True, drop_last=True):
    dataset = DiffDataset(ds_name, split)
    torch_ds = TorchDataset(dataset, max_length=args.max_input,
        img0_transform=img_transform, img1_transform=img_transform
    )

    print("The size of data split %s is %d" % (split, len(torch_ds)))
    loader = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True,
        drop_last=drop_last)
    return dataset, torch_ds, loader

if 'speaker' in args.train:
    train_tuple = get_tuple(args.dataset, 'train', shuffle=True, drop_last=False)
    valid_tuple = get_tuple(args.dataset, 'valid_seen', shuffle=False, drop_last=False)
    speaker = Speaker(train_tuple[0])   

    if args.load is not None:
        print("Load speaker from %s." % args.load)
        speaker.load(args.load)

        scores, result = speaker.evaluate(valid_tuple, split="val_seen")
        print("Have result for %d data" % len(result))
        print("The validation result is:")
        print(scores)
    if args.train == 'speaker':
        speaker.train(train_tuple, valid_tuple, args.epochs)
    if args.train == 'rlspeaker':
        speaker.train(train_tuple, valid_tuple, args.epochs, rl=True)
    elif args.train == 'validspeaker':
        scores, result = speaker.evaluate(valid_tuple)
        print(scores)
    elif args.train == 'testspeaker':
        valid_unseen_tuple = get_tuple(args.dataset, 'valid', shuffle=False, drop_last=False)
        scores, result = speaker.evaluate(valid_unseen_tuple, split="val_unseen")
        print(scores)
        
        # test_tuple = get_tuple(args.dataset, 'test', shuffle=False, drop_last=False)
        # scores, result = speaker.evaluate(test_tuple, split="test_unseen")
        # print("Test:")
        # print("Have result for %d data" % len(result))
        # print(scores)

