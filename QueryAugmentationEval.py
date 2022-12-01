
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
from eval import GlobalDataset,qrels2Evaluator,loadEvalRanklist,DualEncoderRanklist
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
    # AggResults["iterations"].append(0)
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