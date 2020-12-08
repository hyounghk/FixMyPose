import torch
import json
import os
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
from model import Encoder
from decoders import Decoder
from tensorboardX import SummaryWriter
from param import args
import numpy as np
from evaluate import LangEvaluator
import utils
from tqdm import tqdm



rank_099 = torch.tensor(list(range(10))).long().cuda()
def scores_to_ranks(scores):

    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        ranks[i][ranked_idx[i]] = rank_099

    ranks += 1
    return ranks

class Speaker:
    def __init__(self, dataset):

        self.tok = dataset.tok
        self.feature_size = 2048

        self.encoder = Encoder(self.feature_size).cuda()
        ctx_size = self.encoder.ctx_dim
        self.decoder = Decoder(self.tok.vocab_size, ctx_size).cuda()

        # Optimizer
        self.optim = args.optimizer(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                    lr=args.lr)


        # Logs
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.output)     # Tensorboard summary writer

        # Loss
        self.softmax_loss = torch.nn.CrossEntropyLoss()

    def train(self, train_tuple, eval_tuple, num_epochs, rl=False):
        train_ds, train_tds, train_loader = train_tuple
        best_eval_score = 0
        best_epoch = 0

        dist_trg = 0
        dist_src = 0
        tot_num = 0

        train_evaluator = LangEvaluator(train_ds)
        reward_func = lambda uidXpred: train_evaluator.get_reward(uidXpred, args.metric)

        for epoch in range(num_epochs):
            print()
            iterator = tqdm(enumerate(train_loader), total=len(train_tds)//args.batch_size, unit="batch")
            word_accu = 0.

            tot_sum = 0.0
            tot_num = 0.0
            for i, (uid, src, trg, inst, leng, ans_id) in iterator:
                inst = utils.cut_inst_with_leng(inst, leng)
                src, inst = src.cuda(), inst.cuda()
                trg = [t.cuda() for t in trg]
                self.optim.zero_grad()
                
                loss, logits = self.teacher_forcing(src, trg, inst, leng, ans_id, train=True)

                iterator.set_postfix(loss=loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.parameters(), 5.)
                self.optim.step()


                tot_num += logits.size(0)
                max_id = torch.max(logits, dim=-1)[1]
                tot_sum += (max_id == ans_id.cuda()).float().sum()


            scores = tot_sum / tot_num

            print("eval train score", scores)
            if epoch % 1 == 0:
                print("Epoch %d" % epoch, "Best Epoch %d" % best_epoch)

                scores = self.evaluate(eval_tuple)


                print("eval val score", scores)
                if scores > best_eval_score:
                    best_eval_score = scores
                    best_epoch = epoch
                    self.save("best_eval")



    def teacher_forcing(self, src, trg, inst, leng, ans_id, train=True):

        if train:
            self.encoder.train()
            self.encoder.resnet_extractor.eval()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.encoder.resnet_extractor.eval()
            self.decoder.eval()

        # Encoder
        ctx = self.encoder(src, trg)

        # Decoder
        batch_size = inst.size(0)
        scores = self.decoder(inst, leng, ctx, None)


        loss = self.softmax_loss(scores, ans_id.cuda())


        return loss, scores


    def evaluate(self, eval_tuple, iters=-1):
        dataset, th_dset, dataloader = eval_tuple
        evaluator = LangEvaluator(dataset)

        self.encoder.eval()
        self.encoder.resnet_extractor.eval()
        self.decoder.eval()

        all_insts = []
        all_gts = []
        word_accu = 0.

        dist = 0
        tot_sum = 0.0
        tot_num = 0.0
        tot_sum_1rank = 0.0
        tot_sum_3rank = 0.0
        tot_sum_5rank = 0.0
        for i, (uid, src, trg, inst, leng, ans_id) in enumerate(dataloader):
            if i == iters:
                break
            src = src.cuda()
            trg = [t.cuda() for t in trg]

            inst = inst.cuda()
            _, logits = self.teacher_forcing(src, trg, inst, leng, ans_id, train=False)

            bsz = logits.size(0)
            tot_num += bsz
            max_id = torch.max(logits, dim=-1)[1]
            tot_sum += (max_id == ans_id.cuda()).float().sum()


            rank = scores_to_ranks(logits)


            tot_sum_1rank += (rank[range(bsz), ans_id.cuda()] <=1).float().sum()
            tot_sum_3rank += (rank[range(bsz), ans_id.cuda()] <=3).float().sum()
            tot_sum_5rank += (rank[range(bsz), ans_id.cuda()] <=5).float().sum()


        scores = tot_sum / tot_num
        scores_1rank = tot_sum_1rank / tot_num
        scores_3rank = tot_sum_3rank / tot_num
        scores_5rank = tot_sum_5rank / tot_num

        print("eval recall 1:", scores_1rank)
        print("eval recall 3:", scores_3rank)
        print("eval recall 5:", scores_5rank)
        return scores

    def save(self, name):
        encoder_path = os.path.join(self.output, '%s_enc.pth' % name)
        decoder_path = os.path.join(self.output, '%s_dec.pth' % name)
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, path):
        print("Load Speaker from %s" % path)
        enc_path = os.path.join(path + "_enc.pth")
        dec_path = os.path.join(path + "_dec.pth")
        enc_state_dict = torch.load(enc_path)
        dec_state_dict = torch.load(dec_path)
        self.encoder.load_state_dict(enc_state_dict)
        self.decoder.load_state_dict(dec_state_dict)
