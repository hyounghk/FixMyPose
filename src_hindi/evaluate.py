import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'langeval'))
from eval import LanguageEval


class LangEvaluator():
    def __init__(self, dataset):
        self.uid2ref = {}
        self.langeval = LanguageEval()
        for datum in dataset.data:
            self.uid2ref[datum['uid']] = datum['sents_hindi'] 
            
    def evaluate(self, uid2pred, split="val_seen"):
        gts = []
        preds = []
        uids = []
        gen_captions = {}
        for uid, pred in uid2pred.items():
            preds.append(pred)
            gts.append(self.uid2ref[uid])

            uids.append(uid)
            gen_captions[uid] = (pred,self.uid2ref[uid])

        with open("output_" + split + ".json", 'w') as jf:
            json.dump(gen_captions, jf, indent=4, sort_keys=True)
            
        return self.langeval.eval_whole(gts, preds, uids, no_metrics={})

    def get_reward(self, uidXpred, metric="CIDEr"):
        gts = []
        preds = []
        for uid, pred in uidXpred:
            preds.append(pred)
            gts.append(self.uid2ref[uid])
        return self.langeval.eval_batch(gts, preds, metric)


