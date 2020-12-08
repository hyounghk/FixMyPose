from utils import LinearAct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import math
from param import args



class Decoder(nn.Module):
    def __init__(self, ntoken, ctx_size):
        super().__init__()
        hidden_size = args.hid_dim
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.emb_dim = args.emb_dim

        self.img_fc = LinearAct(ctx_size, hidden_size, 'tanh')
        self.w_emb = nn.Embedding(ntoken, self.emb_dim)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(self.emb_dim, hidden_size, batch_first=True)

        self.projection = LinearAct(hidden_size, ntoken)
        self.projection_x = LinearAct(hidden_size*5, hidden_size)

        self.lang_int = LinearAct(hidden_size * 3, hidden_size)
        self.vis_int = LinearAct(hidden_size * 3, hidden_size)

    def forward(self, words, h0, c0, ctx, ctx_mask=None):

        ctx = [self.img_fc(c) for c in ctx]
        src, trg = ctx

        embeds = self.w_emb(words)   

        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        src_att, trg_att = self.attention_cross(src, trg, None)

        src_att = self.drop(src_att)
        trg_att = self.drop(trg_att)

        lnag_src_att = self.attention(x, src_att, None)
        lnag_trg_att = self.attention(x, trg_att, None)


        x = torch.cat((lnag_src_att, lnag_trg_att, x, x*lnag_src_att, x*lnag_trg_att), -1)
        x = self.projection_x(x)

        logit = self.projection(x)

        return logit, h1, c1


    def attention_cross(self, insts_enc, visual, insts_mask):

        sim = torch.matmul(insts_enc, visual.transpose(1,2))

        sim_v = torch.softmax(sim, dim=-1) 

        if insts_mask is not None:
            sim = mask_logits(sim, insts_mask.unsqueeze(-1))   
        sim_l = torch.softmax(sim, dim=1) 

        ltv = torch.matmul(sim_v, visual) 
        vtl = torch.matmul(sim_l.transpose(1,2), insts_enc)

        inst_new = self.lang_int(torch.cat([insts_enc, ltv, insts_enc*ltv], dim=-1))
        vis_new = self.vis_int(torch.cat([visual, vtl, visual*vtl], dim=-1))


        return inst_new, vis_new

    def attention(self, h_att, insts_enc, insts_mask):

        sim = torch.matmul(h_att, insts_enc.transpose(1,2))

        if insts_mask is not None:
            sim = mask_logits(sim, insts_mask.unsqueeze(1))

        h_sofm = torch.softmax(sim, dim=-1) 
        att = torch.matmul(h_sofm, insts_enc) 

        return att
