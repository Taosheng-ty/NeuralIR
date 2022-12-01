"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performances metrices for cossine similarity & dot-product.

Usage:
python eval_msmarco.py model_name [max_corpus_size_in_thousands]
"""

from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models
import logging
import sys
import os
import tarfile
import argparse
from utils.data import loadMSCorpus,loadDevMSqueries,loadDevRelMSQrels
import itertools
from collections import defaultdict
import pytrec_eval
from tqdm import tqdm
from utils.ranking import BM25FirstPhase
import numpy as np
import argparse
import torch
import random
import json
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

GlobalDataset={"DL19":{"testQuery-Passage":"/raid/datasets/shared/MSMARCO_DL19Pass/msmarco-passagetest2019-top1000.tsv", 
                "testQrels":"/raid/datasets/shared/MSMARCO_DL19Pass/2019qrels-pass.txt"},
        "DL20":{"testQuery-Passage":"/raid/datasets/shared/MSMARCO_DL20Pass/msmarco-passagetest2020-top1000.tsv",
                "testQrels":"/raid/datasets/shared/MSMARCO_DL20Pass/2020qrels-pass.txt"},
        "MSDev":{"testQuery-Passage":"/raid/datasets/shared/MSMARCO/top1000.dev.tsv",
                "testQrels":"/raid/datasets/shared/MSMARCO/qrels.dev.tsv"}
        }
BeirDataStorePath="/home/taoyang/.cache/BeirDatasets/"
datasetsName = ["scifact","nfcorpus","fiqa","arguana","scidocs"]
# datasetsName = ["scifact"]
BM25ResultsPath=BeirDataStorePath+"BM25InitialRnk/"
for datasetCurName in datasetsName:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(datasetCurName)
    
    DataDir = os.path.join(BeirDataStorePath, datasetCurName)
    if not os.path.exists(DataDir):
        data_path = util.download_and_unzip(url, BeirDataStorePath)
    GlobalDataset[datasetCurName]={}
    GlobalDataset[datasetCurName]["isBeir"]=True
    GlobalDataset[datasetCurName]["dataPath"]=DataDir
    DatasetBM25Path=os.path.join(BM25ResultsPath,datasetCurName)
#     TestQuery_Passage=os.path.join(DatasetBM25Path,"BM25Top1000.tsv")
#     DevQuery_Passage=os.path.join(DatasetBM25Path,"BM25Top1000.tsv")
    splits=["test"]
    for split in splits:
        splitPath=os.path.join(DatasetBM25Path,split)
        os.makedirs(splitPath, exist_ok=True)
        BM25Path=os.path.join(splitPath,"top1000.tsv")
        QresPath=os.path.join(splitPath,"qrels.tsv")
        if not os.path.exists(BM25Path) or not os.path.exists(QresPath): 
            BM25FirstPhase(DataDir,BM25Path,QresPath,split=split)
        GlobalDataset[datasetCurName][split+"Query-Passage"]=BM25Path
        GlobalDataset[datasetCurName][split+"Qrels"]=QresPath

def LoadMSDevEvaluator(data_folder,Sizelimit=None,*args, **kwargs):
    ### Data files
    
    os.makedirs(data_folder, exist_ok=True)
    ### Load data

    corpus = {}             #Our corpus pid => passage
    dev_queries = {}        #Our dev queries. qid => query
    dev_rel_docs = {}       #Mapping qid => set with relevant pids

    dev_queries=loadDevMSqueries(data_folder)
    dev_rel_docs=loadDevRelMSQrels(data_folder,dev_queries)
    # Read passages
    corpus=loadMSCorpus(data_folder)
    
    if Sizelimit is not None:
        neededPid=list(itertools.chain.from_iterable(dev_rel_docs.values()))
        corpusNew={pid: corpus[pid] for pid in neededPid}
        sizeCur=0
        for pid in corpus:
            if pid in corpusNew:
                continue
            corpusNew[pid]=corpus[pid]
            sizeCur=len(corpusNew.keys())
            if sizeCur>Sizelimit:
                corpus=corpusNew
                logging.info("Eval Corpus: {}".format(len(corpus)))
                break
            
    ## Run evaluator
    logging.info("Queries: {}".format(len(dev_queries)))
    logging.info("Corpus: {}".format(len(corpus)))

    ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs,
                                                            show_progress_bar=True,
                                                            corpus_chunk_size=100000,
                                                            precision_recall_at_k=[10, 100],
                                                            name="msmarco dev",
                                                            *args, **kwargs)
    return ir_evaluator


def qrels2Evaluator(dataset,metrics= pytrec_eval.supported_measures):
    #Read which passages are relevant
    relevant_docs = defaultdict(lambda: defaultdict(int))
    # qrels_filepath = os.path.join(data_folder, '2019qrels-pass.txt')

    # if not os.path.exists(qrels_filepath):
    #     logging.info("Download "+os.path.basename(qrels_filepath))
    #     util.http_get('https://trec.nist.gov/data/deep/2019qrels-pass.txt', qrels_filepath)

    qrels_filepath=GlobalDataset[dataset]["testQrels"]
    with open(qrels_filepath) as fIn:
        for line in fIn:
            qid, _, pid, score = line.strip().split()
            score = int(score)
            if score > 0:
                relevant_docs[qid][pid] = score
    evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, metrics)
    return evaluator
def loadEvalRanklist(dataset):
    queries_passage_filepath=GlobalDataset[dataset]["testQuery-Passage"]
    candidateSet= defaultdict(list)
    num_lines = sum(1 for line in open(queries_passage_filepath))
    n=1
    Corpus=defaultdict(str)
    queries=defaultdict(str)
    with open(queries_passage_filepath, 'r', encoding='utf8') as fIn:
        for line in tqdm(fIn, desc ="Loading data",total=num_lines):
            qid, pid, query,passage = line.strip().split("\t")
            candidateSet[qid].append(pid)
            queries[qid]=query
            Corpus[pid]=passage
    return candidateSet,queries,Corpus
            #  {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}
#     return candidateSet
def DualEncoderRanklist(model,candidateSet,queries,Corpus,batch_size=32):
    docEmb=model.encode(list(Corpus.values()),batch_size=batch_size,show_progress_bar=True)  # encode sentence
    queryEmb=model.encode(list(queries.values()),batch_size=batch_size,show_progress_bar=True)
    queries = dict(zip(list(queries.keys()),queryEmb))
    docs=dict(zip(list(Corpus.keys()),docEmb))
    run={}
    for qid in candidateSet:
        run[qid]={}
        for pid in candidateSet[qid]:
            score=np.sum(docs[pid]*queries[qid])
            run[qid][pid]=float(score)
    return run
            
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='distilbert-base-uncased')
    parser.add_argument("--log_dir",default="output/log", help="where to store the model")
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
    # model_name="msmarco-distilbert-base-tas-b"
    # model_name="output/mse-huggingfaceHard10EpochDist/171600"
    # model_name="../output/log/0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = SentenceTransformer(model_name)
    model.to(model._target_device)
    model.eval()
    data_folder =os.path.join(os.path.expanduser('~'), '.cache/MSMARCO')
    os.makedirs(data_folder,exist_ok=True)
    AggResults= defaultdict(list)
    # ir_evaluator=LoadMSDevEvaluator(data_folder,Sizelimit=1000)
    # ir_evaluator=LoadMSDevEvaluator(data_folder)
    # retrieResult=ir_evaluator.compute_metrices(model)
    # print(ir_evaluator(model))
    # AggResults["MSMARCOPassDot"].append(retrieResult["dot_score"]["ndcg@k"])
    # AggResults["MSMARCOPassCos"].append(retrieResult["cos_sim"]["ndcg@k"])
    AggResults["iterations"].append(0)
    dataNames=list(GlobalDataset.keys())[:2]
    # dataNames=list(GlobalDataset.keys())
    batch_size=128
    # dataNames=["scifact"]
    for dataName in  dataNames:
        evaluator=qrels2Evaluator(dataName,metrics={'ndcg_cut.10'})
        candidateSet,queries,Corpus=loadEvalRanklist(dataName)
        
        run=DualEncoderRanklist(model,candidateSet,queries,Corpus,batch_size=batch_size)
        EvalResults=evaluator.evaluate(run)

        RealCalMetrics=list(EvalResults.values())[0].keys()
        for measure in sorted(RealCalMetrics):
            AggResults[dataName+measure].append(pytrec_eval.compute_aggregated_measure(
                    measure,
                    [query_measures[measure]
                        for query_measures in EvalResults.values()]))
    print(AggResults)
    with open(args.log_dir+"/AggResults.jjson", "w") as outfile:
        # outfile.write(ending)
        json.dump(AggResults,outfile)  
    
    