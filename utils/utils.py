from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models, losses, InputExample
import numpy as np
import os
import logging
import tarfile
from torch.utils.data import Dataset,DataLoader
from torch import nn, Tensor
import torch
from tqdm import tqdm
from sentence_transformers.util import cos_sim, dot_score
from sentence_transformers.util import import_from_string, batch_to_device, fullname, snapshot_download

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def loadEvalData(data_folder,corpus_max_size=np.inf):
    corpus = {}             #Our corpus pid => passage
    dev_queries = {}        #Our dev queries. qid => query
    dev_rel_docs = {}       #Mapping qid => set with relevant pids
    needed_pids = set()     #Passage IDs we need
    needed_qids = set()     #Query IDs we need
    os.makedirs(data_folder, exist_ok=True)
    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    dev_queries_file = os.path.join(data_folder, 'queries.dev.small.tsv')
    qrels_filepath = os.path.join(data_folder, 'qrels.dev.tsv')
    # Load the 6980 dev queries
    # query_max_size=100
    ### Download files if needed
    if not os.path.exists(collection_filepath) or not os.path.exists(dev_queries_file):
        tar_filepath = os.path.join(data_folder, 'collectionandqueries.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download: "+tar_filepath)
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)


    if not os.path.exists(qrels_filepath):
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv', qrels_filepath)
    with open(dev_queries_file, encoding='utf8') as fIn:
        for line in fIn:
            # if len(dev_queries) > query_max_size:
            #     break
            qid, query = line.strip().split("\t")
            dev_queries[qid] = query.strip()


    # Load which passages are relevant for which queries
    with open(qrels_filepath) as fIn:
        for line in fIn:
            qid, _, pid, _ = line.strip().split('\t')

            if qid not in dev_queries:
                continue

            if qid not in dev_rel_docs:
                dev_rel_docs[qid] = set()
            dev_rel_docs[qid].add(pid)

            needed_pids.add(pid)
            needed_qids.add(qid)
    # Read passages
    with open(collection_filepath, encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            passage = passage

            if pid in needed_pids or corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
                corpus[pid] = passage.strip()
            # if len(corpus) > corpus_max_size:
            #     break
    return dev_queries,dev_rel_docs,corpus
# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, triplet):
        self.triplet = triplet
        # self.queries
        # for triplet in self.triplet:
        #     self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
        #     self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
        #     random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        # query = self.queries[self.queries_ids[item]]
        # query_text = query['query']

        # pos_id = query['pos'].pop(0)    #Pop positive and add at end
        # pos_text = self.corpus[pos_id]
        # query['pos'].append(pos_id)

        # neg_id = query['neg'].pop(0)    #Pop negative and add at end
        # neg_text = self.corpus[neg_id]
        # query['neg'].append(neg_id)
        query_text, pos_text, neg_text=self.triplet[item]
        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.triplet)

class MSMARCODatasetCross(Dataset):
    def __init__(self, train_samples):
        self.train_samples = train_samples
        # self.queries
        # for triplet in self.triplet:
        #     self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
        #     self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
        #     random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        # query = self.queries[self.queries_ids[item]]
        # query_text = query['query']

        # pos_id = query['pos'].pop(0)    #Pop positive and add at end
        # pos_text = self.corpus[pos_id]
        # query['pos'].append(pos_id)

        # neg_id = query['neg'].pop(0)    #Pop negative and add at end
        # neg_text = self.corpus[neg_id]
        # query['neg'].append(neg_id)
        query_text, passage_text, label=self.train_samples[item]
        return InputExample(texts=[query_text, passage_text],label=label)

    def __len__(self):
        return len(self.train_samples)


class EntropyLoss(nn.Module):
    def __init__(self,model):
        super(EntropyLoss, self).__init__()
        self.model=model.model
        self.cross_entropy_loss=nn.CrossEntropyLoss()
    def forward(self,sentence_features, *args, **kwargs):
        reps=[self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        qemb=reps[0]
        demb=torch.cat(reps[1:])
        cosSimi=cos_sim(qemb,demb)
        numBatch=cosSimi.shape[0]
        # label=torch.arange(numBatch)
        label = torch.tensor(range(len(cosSimi)), dtype=torch.long, device=cosSimi.device)  # Example a[i] should match with b[i]
        loss=self.cross_entropy_loss(cosSimi,label)
        return loss
class CrossBCELoss(nn.Module):
    def __init__(self,model):
        super(CrossBCELoss, self).__init__()
        self.model=model
        self.loss=nn.BCELoss()
        self.actfn = nn. Sigmoid()
    def forward(self,sentence_features, label, **kwargs):
        logits=self.model(**sentence_features, return_dict=True)
        logits=self.actfn(logits.logits)
        logits = logits.view(-1)
        loss=self.loss(logits,label)
        return loss
def train_one_epoch(epoch_index, tb_writer,training_loader,loss_fn,optimizer,model):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(tqdm(training_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        # labels =labels.to(model._target_device)
        # inputs = list(map(lambda batch: batch_to_device(batch, model._target_device), inputs))
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        # outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(inputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    del inputs, labels
    return last_loss
### test
if __name__ == "__main__":
    data_folder = '/raid/datasets/shared/MSMARCO'
    corpus_max_size=10**4
    dev_queries,dev_rel_docs,corpus=loadEvalData(data_folder,corpus_max_size=corpus_max_size)