import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from utils import LinearAct
import math
from param import args



class Encoder(nn.Module):
    def __init__(self, feature_size, normalized=False):
        super(Encoder, self).__init__()
        self.feature_size = feature_size

        # ResNet Feat Mean and Std
        self.normazlied = normalized
        if normalized:
            import numpy as np
            import os
            DATA_ROOT = "dataset/"
            feat_mean = np.load(os.path.join(DATA_ROOT, args.dataset, 'feat_mean.npy'))
            feat_std = np.load(os.path.join(DATA_ROOT, args.dataset, 'feat_std.npy'))
            self.feat_mean = torch.from_numpy(feat_mean).cuda()
            self.feat_std = torch.from_numpy(feat_std).cuda()

        # Resnet Feature Extractor
        resnet_extractor = models.resnet101(pretrained=True)
        modules = list(resnet_extractor.children())[:-3]
        self.resnet_extractor = nn.Sequential(*modules)
        for p in self.resnet_extractor.parameters():
            p.requires_grad = False

    @property
    def ctx_dim(self):
        return self.feature_size

    def forward(self, src, trg):
        """
        :param src: src_image
        :param trg: trg_image
        :return: ctx (whatever it is)
        """
        if args.img_type == 'feat':
            src_feat, trg_feat = src, trg
        else:
            # Feature Extraction
            src_feat = self.resnet_extractor(src)
            trg_feat = [self.resnet_extractor(t) for t in trg]

            # Shape
            src_feat = src_feat.permute(0, 2, 3, 1)     # N, C, H, W --> N, H, W, C
            # trg_feat = trg_feat.permute(0, 2, 3, 1)
            trg_feat = [t.permute(0, 2, 3, 1) for t in trg_feat]

        src_feat = src_feat.view(src_feat.size(0), -1, src_feat.size(-1))
        trg_feat = [t.view(t.size(0), -1, t.size(-1)) for t in trg_feat]

        # normalize
        if self.normazlied:
            src_feat = (src_feat - self.feat_mean) / self.feat_std
            trg_feat = (trg_feat - self.feat_mean) / self.feat_std

        # tuple
        ctx = (src_feat, trg_feat)

        return ctx
