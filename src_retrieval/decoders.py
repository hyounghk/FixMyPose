from utils import LinearAct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import math
from param import args
import numpy as np

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)



class Decoder(nn.Module):
    def __init__(self, ntoken, ctx_size):
        super().__init__()
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.emb_dim = args.emb_dim

        self.ntoken = ntoken

        self.img_fc = LinearAct(1024, hidden_size, 'tanh')
        self.w_emb = nn.Embedding(ntoken, self.emb_dim)
        self.drop = nn.Dropout(0.5)

        self.lstm = nn.LSTM(self.emb_dim, hidden_size, bidirectional=True, batch_first=True)

        self.projection = LinearAct(hidden_size, 1)
        self.projection_lang = LinearAct(hidden_size * 3, hidden_size)

        self.projection_lang_vis = LinearAct(hidden_size * 3, hidden_size)
        self.projection_vis = LinearAct(hidden_size * 3, hidden_size)
        self.projection_att = LinearAct(hidden_size * 3, 1)

        self.lang_int_lang = LinearAct(hidden_size * 3, hidden_size)
        self.vis_int_lang = LinearAct(hidden_size * 3, hidden_size)

        self.lang_projection = LinearAct(hidden_size*2, hidden_size, 'tanh')


    def init_emb(self, np_file, requires_grad=False):
        weight_init = torch.from_numpy(np.load(np_file))
        print(weight_init.shape)
        print(self.ntoken, self.emb_dim)
        assert weight_init.shape == (self.ntoken, self.emb_dim)

        self.w_emb.weight.data[4:self.ntoken] = weight_init[4:self.ntoken]
        self.w_emb.weight.requires_grad = requires_grad

    def forward(self, words, leng, ctx,ctx_mask=None):

        
        bsz = words.size(0)
        
        src = self.img_fc(ctx[0])

        trgs = [self.img_fc(c) for c in ctx[1]]

        embeds = self.w_emb(words)      # batch_size, length, emb_dim
        embeds = self.drop(embeds)

        with torch.no_grad():
            insts_mask = torch.zeros(bsz, words.size(1)).cuda()

        for idx in range(bsz):
            insts_mask[idx,:leng[idx]] = 1

        # LSTM
        hid_dim = src.size(-1)
        h0 = torch.zeros(2, bsz, hid_dim).cuda()
        c0 = torch.zeros(2, bsz, hid_dim).cuda()

        self.lstm.flatten_parameters()
        inst, (h1, c1) = self.lstm(embeds, (h0, c0))
        inst = self.lang_projection(inst)
        # inst = self.drop(inst)
        scores = []

        for i, trg in enumerate(trgs):


            inst_src_att, src_lang_att = self.attention_cross_lang(inst, src, insts_mask)
            inst_trg_att, trg_lang_att = self.attention_cross_lang(inst, trg, insts_mask)
            inst_src_att = self.drop(inst_src_att)
            inst_trg_att = self.drop(inst_trg_att)
            src_lang_att = self.drop(src_lang_att)
            trg_lang_att = self.drop(trg_lang_att)

            inst_src_1, inst_trg_1 = self.attention_cross_lang(inst_src_att, trg_lang_att, insts_mask)
            src_inst_1, trg_inst_1 = self.attention_cross_lang(inst_trg_att, src_lang_att, insts_mask)

            src_inst_1 = self.drop(src_inst_1)
            trg_inst_1 = self.drop(trg_inst_1)

            feats_inst = torch.cat([inst_src_1, src_inst_1, inst_src_1*src_inst_1], dim=-1)
            feats_vis = torch.cat([inst_trg_1, trg_inst_1, inst_trg_1*trg_inst_1], dim=-1)

            feats_inst_w = self.projection_lang(self.weighedsum(feats_inst, insts_mask))
            feats_vis_w = self.projection_vis(self.weighedsum(feats_vis, None))

            feats_vis_inst = self.projection_lang_vis(torch.cat([feats_inst_w, feats_vis_w,feats_inst_w*feats_vis_w], dim=-1))

            logit = self.projection(feats_vis_inst)
            
            scores.append(logit)


        scores = torch.stack(scores, dim=1).view(-1, 10)

        return scores

    def weighedsum(self, feats, mask):

        feats_pro = self.projection_att(feats)
        if mask is not None:
            feats_pro = mask_logits(feats_pro, mask.unsqueeze(-1))

        feats_sm = torch.softmax(feats_pro, dim=1)

        feats_weited_sum = (feats * feats_sm).sum(1)

        return feats_weited_sum


    def attention_cross_lang(self, insts_enc, visual, insts_mask):

        sim = torch.matmul(insts_enc, visual.transpose(1,2)) #(N, L, D)

        sim_v = torch.softmax(sim, dim=-1) #(N, L, V)

        if insts_mask is not None:
            sim = mask_logits(sim, insts_mask.unsqueeze(-1))   
        sim_l = torch.softmax(sim, dim=1) #(N, L, V)

        ltv = torch.matmul(sim_v, visual) #(N, L, D)
        vtl = torch.matmul(sim_l.transpose(1,2), insts_enc) #(N, V, D)

        inst_new = self.lang_int_lang(torch.cat([insts_enc, ltv, insts_enc*ltv], dim=-1))
        vis_new = self.vis_int_lang(torch.cat([visual, vtl, visual*vtl], dim=-1))


        return inst_new, vis_new


