"""
revised from sentence bert msmarco training examples.
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus

Running this script:
python train_bi-encoder-v3.py
"""
from torch.utils.data import DataLoader
from sentence_transformers import losses
import logging
import os
import argparse
from utils.model import loadModel
from utils.utils import logging
from utils.data import getTriplet
import torch
import random
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", default='distilbert-base-uncased')
parser.add_argument("--log_dir",default="output/log", help="where to store the model")
parser.add_argument("--loss", default='mnrl', type=str)
parser.add_argument("--tripleSource", default='huggingfaceHard',choices=["RandomNeg","huggingfaceHard"], type=str)
# parser.add_argument("--negs_to_use", default=[],choices=[["BM25"],[]], type=list)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--ce_score_margin", default=3.0, type=float)
parser.add_argument("--gpu", type=int, default=None, nargs="+", help="used gpu")
args = parser.parse_args()
if args.gpu is None:
    devices=list(range(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(random.choice(devices))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print(args)
# The  model we want to fine-tune
model_name = args.model_name
negs_to_use=args.negs_to_use.split(",") if args.negs_to_use is not None else []
train_batch_size = args.train_batch_size           #Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory
ce_score_margin = args.ce_score_margin             #Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = args.num_negs_per_system         # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs                 # Number of epochs we want to train
settingDict = vars(args)
model=loadModel(**settingDict)


os.makedirs(args.log_dir,exist_ok=True)
model_save_path=args.log_dir

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCO')
# data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCOToy')
os.makedirs(data_folder,exist_ok=True)
# corpus,queries,train_triplet=loadData(data_folder)
# logging.info("Train queries: {}".format(len(train_triplet)))
# train_dataset = MSMARCODataset(train_triplet, corpus=corpus)

train_dataset=getTriplet(data_folder,source=args.tripleSource,negs_to_use = negs_to_use,num_negs_per_system=num_negs_per_system,ce_score_margin=ce_score_margin)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
lossDict={"mnrl":losses.MultipleNegativesRankingLoss,"marginalMSE":losses.MarginMSELoss}
train_loss = lossDict[args.loss](model=model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=len(train_dataloader),
          optimizer_params = {'lr': args.lr},
          )
# Save the model
model.save(model_save_path)
json_object=""
with open(model_save_path+"/sample.jjson", "w") as outfile:
    outfile.write(json_object)