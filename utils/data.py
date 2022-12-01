
import os
import sys
from sentence_transformers import  LoggingHandler
# scriptPath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(scriptPath)
# print(scriptPath)
# from utils.utils import logging
# from utils import logging
import logging
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
import tarfile
import pickle
import gzip
import tqdm
import json
import  sentence_transformers 
from sentence_transformers import InputExample
from torch.utils.data import Dataset
import random
def loadMSCorpus(data_folder,*args, **kwargs):
    #### Read the corpus files, that contain all the passages. Store them in the corpus dict
    corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    collection_picked_file= os.path.join(data_folder, 'collection.pickle')
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download collection.tar.gz")
            sentence_transformers.util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    logging.info("Read corpus: collection.tsv")

    if os.path.exists(collection_picked_file):
        with open(collection_picked_file, 'rb') as f:
            corpus = pickle.load(f)
    else:
        num_lines = sum(1 for line in open(collection_filepath))
        with open(collection_filepath, 'r', encoding='utf8') as fIn:
            for line in tqdm.tqdm(fIn, desc ="Loading data",total=num_lines):
                pid, passage = line.strip().split("\t")
                pid = int(pid)
                corpus[pid] = passage
            with open(collection_picked_file, "wb") as myFile:
                pickle.dump(corpus,myFile)
    return corpus

def loadTrainMSqueries(data_folder,*args, **kwargs):
    ### Read the train queries, store in queries dict
    queries = {}        #dict in the format: query_id -> query. Stores all training queries
    logging.info("Read queries: queries.train.tsv")
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    queries_filepath_picked_file= os.path.join(data_folder, 'queries.train.tsv.pickle')
    if not os.path.exists(queries_filepath_picked_file):
        if not os.path.exists(queries_filepath):
            tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
            if not os.path.exists(tar_filepath):
                logging.info("Download queries.tar.gz")
                sentence_transformers.util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)
            with tarfile.open(tar_filepath, "r:gz") as tar:
                tar.extractall(path=data_folder)
        with open(queries_filepath, 'r', encoding='utf8') as fIn:
            for line in fIn:
                qid, query = line.strip().split("\t")
                qid = int(qid)
                queries[qid] = query
        with open(queries_filepath_picked_file, "wb") as myFile:
            pickle.dump(queries,myFile)
    else:
        with open(queries_filepath_picked_file, 'rb') as f:
            queries = pickle.load(f)
    return queries

def loadDevMSqueries(data_folder,reload=False,*args, **kwargs):
    ### Read the train queries, store in queries dict
    queries = {}        #dict in the format: query_id -> query. Stores all training queries
    logging.info("Read queries: queries.dev.tsv")
    queries_filepath = os.path.join(data_folder, 'queries.dev.small.tsv')
    queries_filepath_picked_file= os.path.join(data_folder, 'queries.dev.small.tsv.pickle')
    if not os.path.exists(queries_filepath_picked_file) or reload:
        if not os.path.exists(queries_filepath):
            if not os.path.exists(queries_filepath):
                tar_filepath = os.path.join(data_folder, 'collectionandqueries.tar.gz')
                sentence_transformers.util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz', tar_filepath)
                with tarfile.open(tar_filepath, "r:gz") as tar:
                    tar.extractall(path=data_folder)
        with open(queries_filepath, 'r', encoding='utf8') as fIn:
            for line in fIn:
                qid, query = line.strip().split("\t")
                qid = int(qid)
                queries[qid] = query
        with open(queries_filepath_picked_file, "wb") as myFile:
            pickle.dump(queries,myFile)
    else:
        with open(queries_filepath_picked_file, 'rb') as f:
            queries = pickle.load(f)
    logging.info("Dev query size {}".format(len(queries)))
    return queries

def loadDevRelMSQrels(data_folder,dev_queries=None,reload=False):
    # Load which passages are relevant for which queries
    dev_rel_docs={}
    qrel_filepath = os.path.join(data_folder, 'qrels.dev.tsv')
    qrel_filepath_picked_file= os.path.join(data_folder, 'qrels.dev.tsv.pickle')
    logging.info("Load qrels.dev.tsv")
    if not os.path.exists(qrel_filepath):
        sentence_transformers.util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv', qrel_filepath)
    if not os.path.exists(qrel_filepath_picked_file) or reload:
        with open(qrel_filepath) as fIn:
            for line in fIn:
                qid, _, pid, _ = line.strip().split('\t')
                qid=int(qid)
                pid=int(pid)
                if dev_queries is not None and qid not in dev_queries:
                    continue
                if qid not in dev_rel_docs:
                    dev_rel_docs[qid] = set()
                dev_rel_docs[qid].add(pid)
        with open(qrel_filepath_picked_file, "wb") as myFile:
            pickle.dump(dev_rel_docs,myFile)
    else:
        with open(qrel_filepath_picked_file, 'rb') as f:
            dev_rel_docs = pickle.load(f)
    return dev_rel_docs        

def loadHardNeg(queries,data_folder,ce_score_margin,negs_to_use = [],use_all_queries=False,num_negs_per_system=5,*args, **kwargs):
    # Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
    # to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
    prefix="All" if len(negs_to_use)==0 else "+".join(negs_to_use)
    
    train_queries_picked_file=os.path.join(data_folder, str(num_negs_per_system)+"_"+str(ce_score_margin)+prefix+'train_queries.pickle')
    logging.info("Load Hard Negativ")
    if os.path.exists(train_queries_picked_file):
        with open(train_queries_picked_file, 'rb') as f:
            train_queries = pickle.load(f)
    else:
        ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
        ce_scores_fileBin = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl')
        if not os.path.exists(ce_scores_fileBin):
            if not os.path.exists(ce_scores_file):
                logging.info("Download cross-encoder scores file")
                sentence_transformers.util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)
                logging.info("Load CrossEncoder scores dict")
            with gzip.open(ce_scores_file, 'rb') as fIn:
                ce_scores = pickle.load(fIn)
            with open(ce_scores_fileBin, "wb") as myFile:
                pickle.dump(ce_scores,myFile)
        else:
            with open(ce_scores_fileBin, 'rb') as f:
                ce_scores = pickle.load(f)
        # As training data we use hard-negatives that have been mined using various systems
        hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
        if not os.path.exists(hard_negatives_filepath):
            logging.info("Download hard negative id file")
            sentence_transformers.util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz', hard_negatives_filepath)
        logging.info("Read hard negatives train file")
        train_queries = {}
        num_lines = sum(1 for line in gzip.open(hard_negatives_filepath))
        with gzip.open(hard_negatives_filepath, 'rt') as fIn:
            for line in tqdm.tqdm(fIn,total=num_lines):
                data = json.loads(line)

                #Get the positive passage ids
                qid = data['qid']
                pos_pids = data['pos']

                if len(pos_pids) == 0:  #Skip entries without positives passages
                    continue

                pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
                ce_score_threshold = pos_min_ce_score - ce_score_margin

                #Get the hard negatives
                neg_pids = set()
                if len(negs_to_use)==0 :#Use all systems
                    negs_to_use = list(data['neg'].keys())
                    # if args.negs_to_use is not None:    #Use specific system for negatives
                    #     negs_to_use = args.negs_to_use.split(",")
                # else:   
                #     # negs_to_use = negs_to_use.split(",")
                #     logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

                for system_name in negs_to_use:
                    if system_name not in data['neg']:
                        continue
                    system_negs = data['neg'][system_name]
                    negs_added = 0
                    for pid in system_negs:
                        if ce_scores[qid][pid] > ce_score_threshold:
                            continue

                        if pid not in neg_pids:
                            neg_pids.add(pid)
                            negs_added += 1
                            if negs_added >= num_negs_per_system:
                                break
                if use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
                    train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}
            with open(train_queries_picked_file, "wb") as myFile:
                pickle.dump(train_queries,myFile)
        del ce_scores
    return train_queries

def loadOrigTriplet(data_folder):
    triplet_filepath = os.path.join(data_folder, 'triples.train.small.tsv')
    tar_filepath = os.path.join(data_folder, 'triples.train.small.tar.gz')
    queries_filepath_picked_file= os.path.join(data_folder, 'triples.train.small.tsv.pickle')
    if not os.path.exists(queries_filepath_picked_file):
        if not os.path.exists(tar_filepath):
            logging.info("Download hard negative id file")
            sentence_transformers.util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz', tar_filepath)
            with tarfile.open(tar_filepath, "r:gz") as tar:
                tar.extractall(path=data_folder)
        triplet=[]
        # with open(triplet_filepath, "r") as f:
        #     num_lines= len(f.readlines())
        num_lines=39780811
        with open(triplet_filepath, 'r', encoding='utf8') as fIn:
            for line in tqdm.tqdm(fIn,total=num_lines):
                query, PosPassage,NegPassage = line.strip().split("\t")
                triplet.append([query, PosPassage,NegPassage])
        with open(queries_filepath_picked_file, "wb") as myFile:
            pickle.dump(triplet,myFile)
    else:   
        with open(queries_filepath_picked_file, 'rb') as f:
            triplet = pickle.load(f)
    logging.info("Load {} triplets from {}".format(len(triplet),triplet_filepath))
    return triplet


def loadData(data_folder,ce_score_margin=0.4,*args, **kwargs):
    corpus=loadMSCorpus(data_folder,*args, **kwargs)
    queries=loadTrainMSqueries(data_folder,*args, **kwargs)
    train_queries=loadHardNeg(queries,data_folder,ce_score_margin,*args, **kwargs)
    return corpus,queries,train_queries

# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
def loadcseFiles(data_folder,*args, **kwargs):
    ce_scores_fileBin = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl')
    with open(ce_scores_fileBin, 'rb') as f:
        ce_scores = pickle.load(f)
    return ce_scores

class MSMARCODatasetHuggingfaceHardTriplet(Dataset):
    def __init__(self, queries, corpus,ce_scores):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores=ce_scores
        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        qid=self.queries_ids[item]
        query = self.queries[qid]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)
        pos_score = self.ce_scores[qid][pos_id]
        neg_score = self.ce_scores[qid][neg_id]
        return InputExample(texts=[query_text, pos_text, neg_text],label=pos_score-neg_score)
    def __len__(self):
        return len(self.queries)
class MSMARCODatasetRandomNegTriplet(Dataset):
    def __init__(self, queries, corpus,seed=0):
        random.seed(seed)
        logging.info("Using random triplets")
        self.queries = queries
        self.corpus = corpus
        self.queries_ids = list(queries.keys())
        self.corpus_ids=list(self.corpus.keys())
        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])

    def __getitem__(self, item):
        qid=self.queries_ids[item]
        query = self.queries[qid]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        # neg_id = query['neg'].pop(0)    #Pop negative and add at end
        # neg_text = self.corpus[neg_id]
        # query['neg'].append(neg_id)
        neg_id=random.choice(self.corpus_ids)
        neg_text = self.corpus[neg_id]
        # pos_score = self.ce_scores[qid][pos_id]
        # neg_score = self.ce_scores[qid][neg_id]
        return InputExample(texts=[query_text, pos_text, neg_text])
    def __len__(self):
        return len(self.queries)
class MSMARCODatasetOrigTriplet(Dataset):
    def __init__(self, triplet):
        self.triplet = triplet
    def __getitem__(self, item):
        query_text,pos_text,neg_text=self.triplet[item]
        return InputExample(texts=[query_text, pos_text, neg_text])
    def __len__(self):
        return len(self.triplet)

def getTriplet(data_folder,source="huggingfaceHard",ce_score_margin=0.5,*args, **kwargs):
    if source=="RandomNeg":
        corpus=loadMSCorpus(data_folder,*args, **kwargs)
        queries=loadTrainMSqueries(data_folder,*args, **kwargs)
        train_queries=loadHardNeg(queries,data_folder,ce_score_margin,*args, **kwargs)
        return MSMARCODatasetRandomNegTriplet(train_queries,corpus)
    elif source=="huggingfaceHard":
        corpus=loadMSCorpus(data_folder,*args, **kwargs)
        queries=loadTrainMSqueries(data_folder,*args, **kwargs)
        train_queries=loadHardNeg(queries,data_folder,ce_score_margin,*args, **kwargs)
        ce_scores=loadcseFiles(data_folder,*args, **kwargs)
        return MSMARCODatasetHuggingfaceHardTriplet(train_queries,corpus,ce_scores)
    else:
        raise 
    
def LoadHardNegfromhuggingface(data_folder):
        hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
        if not os.path.exists(hard_negatives_filepath):
            logging.info("Download hard negative id file")
            sentence_transformers.util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz', hard_negatives_filepath)
        logging.info("Read hard negatives train file")
        num_lines = sum(1 for line in gzip.open(hard_negatives_filepath))
        with gzip.open(hard_negatives_filepath, 'rt') as fIn:
            for line in tqdm.tqdm(fIn,total=num_lines):
                data = json.loads(line)
                #Get the positive passage ids
                qid = data['qid']
                pos_pids = data['pos']        
if __name__=="__main__":
    data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCO')
    # data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCOToy')
    os.makedirs(data_folder,exist_ok=True)
    # loadData(data_folder)
    # deq=loadDevMSqueries(data_folder,reload=True)
    # triplet=loadOrigTriplet(data_folder)
    
    # loadDevRelMSQrels(data_folder,deq,reload=True)
    # TripletDataset=getTriplet(data_folder,source="RandomNeg")
    # TripletDataset=LoadHardNegfromhuggingface(data_folder)
    TripletDataset=getTriplet(data_folder,source="huggingfaceHard",negs_to_use=["bm25"])
    # ce_scores=loadcseFiles(data_folder)