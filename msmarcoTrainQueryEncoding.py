from itertools import count
from collections import defaultdict
from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models
import os
from secrets import choice
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import pickle
import os
import random
import torch
import json
from beir.datasets.data_loader import GenericDataLoader
## we will encode passages and pickle it.
parser = argparse.ArgumentParser()
# parser.add_argument("--DataSet", default="scifact",choices=["DL19","DL20","scifact","nfcorpus","fiqa","arguana","scidocs","vihealthqa"], type=str,help="The dataset we want to use, please specify it in config.py")
parser.add_argument("--Model", default='msmarco-bert-base-dot-v5', type=str,help="Here we use the trained models on SentenceTransformers, please refer https://www.sbert.net/docs/pretrained_models.html")
parser.add_argument("--log_dir", default="output/", type=str,help="the output directory")
parser.add_argument("--batch_size", default=32, type=int,help="batch size when encoding the documents")
parser.add_argument("--gpu", type=int, default=None, nargs="+", help="used gpu id")

args = parser.parse_args()

## set the gup accordingly
if args.gpu is None:
    devices=list(range(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(random.choice(devices))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
model = SentenceTransformer(args.Model)
# print(config.dataset)

log_dir=args.log_dir
# docEmb=model.encode(list(docs.values()),batch_size=args.batch_size,show_progress_bar=True)
# os.makedirs(args.log_dir, exist_ok=True)
# pickle.dump([queries,docs], open(args.log_dir+"/encoding.pkl", "wb")) # pickle it.
factory = lambda c=count(): 0
pdict = defaultdict(factory)
qdict = defaultdict(factory)
qRels = defaultdict(list)
data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCO')
qrel_filepath = os.path.join(data_folder, 'qrels.train.tsv')
with open(qrel_filepath) as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip().split('\t')
        pdict[pid]+=1
        qdict[qid]+=1
        qRels[int(qid)].append(int(pid))
        
data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCO')
qrel_filepath = os.path.join(data_folder, 'queries.train.tsv')
queriesText=defaultdict(str)
with open(qrel_filepath) as fIn:
    for line in fIn:
        qid, queries = line.strip().split('\t')
        queriesText[qid]=queries
model_name="msmarco-bert-base-dot-v5"
# model_name="output/mse-huggingfaceHard10EpochDist/171600"
# model_name="../output/log/0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# model = SentenceTransformer(model_name)
model.to(model._target_device)
qEmb=model.encode(list(queriesText.values()),batch_size=32,show_progress_bar=True)
queries = dict(zip(list(queriesText.keys()),qEmb))
os.makedirs(args.log_dir, exist_ok=True)
pickle.dump([queries], open(args.log_dir+"/encoding.pkl", "wb")) # pickle it.
