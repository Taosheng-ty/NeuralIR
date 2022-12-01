import numpy as np
from sentence_transformers import  util
from collections import defaultdict
import pytrec_eval
import json
import os
from rank_bm25 import BM25Okapi
import csv
import numpy as np
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
def getMMRRanklist(queryRep,
                        qPassageRep,
                        risk_preference_param,
                        w_k,
                        **param):
    """_summary_

    Args:
        queryRep (_type_):  query representation
        qPassageRep (_type_):  passages representation
        risk_preference_param (_type_):  risk preference parameter
        w_k (_type_): the weight we put on rank k. e.g., we can use ndcg weight.

    Returns:
        _type_: _description_
    """
    score_qid_mean=util.dot_score(queryRep,qPassageRep)[0]   # get relevance
    score_qid_cov=np.cov(qPassageRep)   # get covariance
    CandidateLength=score_qid_mean.shape[0]
    RankLength=min(len(w_k),CandidateLength)
    ranklist=[]
    for i in range(RankLength):  ## to construct a ranklist, we iteratively select one item at each time.
        SelectedId=getNextMaximalOne(score_qid_mean,
                        score_qid_cov,
                        ranklist,
                        risk_preference_param,
                        w_k)
        ranklist.append(SelectedId)  ## append selected one to ranklist
    return ranklist
def getNextMaximalOne(score_qid_mean,
                        score_qid_cov,
                        selected_item_id,
                        risk_preference_param,
                        w_k,
                        **param):
    """_summary_
    this funciton selects the next one item when constructing a ranklist.
    Plese refer Eq.9 in the paper https://dl.acm.org/doi/pdf/10.1145/1571941.1571963
    Args:
        score_qid_mean (_type_):  numpy array shape=(NumDoc)
        score_qid_cov (_type_):  numpy array shape=(NumDoc,NumDoc)
        selected_item_id (_type_):   already selected items, list  
        risk_preference_param (_type_):   risk preference parameter, i.e. b in eq.9
        w_k (_type_): _description_ : the weight we put on rank k. e.g., we can use ndcg weight. i.e. w_k in eq.9

    Returns:
        _type_: _description_
    """
    if len(selected_item_id)==0:
        return score_qid_mean.argmax()
    numItems=score_qid_mean.shape[0]
    selected_item_id=np.array(selected_item_id)
    Candidate_id=np.delete(np.arange(numItems),selected_item_id)

    current_rank=len(selected_item_id)
    First=score_qid_mean[Candidate_id] #first part E[r_k] in Eq.9 of https://dl.acm.org/doi/pdf/10.1145/1571941.1571963

    Second=-risk_preference_param*w_k[current_rank]*score_qid_cov[Candidate_id,Candidate_id] #second part  -b*w_k*\sigma_k^2 in Eq.9
    Third=-2*risk_preference_param*np.sum(score_qid_cov[Candidate_id[:,None],selected_item_id[None,:]]*w_k[:current_rank],axis=1) # third part -2b*\sum_{i=1}^{k-1} w_i* cov_{i,k} in Eq.9

    marginal_value_cur=First+Second+Third
    SelectedId=Candidate_id[np.argmax(marginal_value_cur)] # And we select the one with maximal marginal gain. 
    return SelectedId

def getCorrRank(queryRep,
                        qPassageRep,
                        risk_preference_param,
                        w_k,
                        rmDiag=True,
                        EvidenceTopk=10,
                        **param):
    """_summary_

    Args:
        queryRep (_type_):  query representation
        qPassageRep (_type_):  passages representation
        risk_preference_param (_type_):  risk preference parameter
        w_k (_type_): from which, we need to know the RankLength.

    Returns:
        _type_: _description_
    """
    QueryPassEmb=np.concatenate([queryRep[None,:],qPassageRep])
    QPcovariance=np.cov(QueryPassEmb)
    QPcovariance=QPcovariance-QPcovariance.mean()
    QPScore=QPcovariance[0,1:]
    CovWeight=np.zeros_like(QPScore)
    CandidateLength=qPassageRep.shape[0]
    EvidenceTopk=min(EvidenceTopk,CandidateLength)
    TopKind = np.argpartition(-QPScore, EvidenceTopk)[:EvidenceTopk]
    CovWeight[TopKind]=QPScore[TopKind]
    CovWeight[QPScore<=0]=0.0
    Pcovaraince=QPcovariance[1:,1:]
    if rmDiag:
        np.fill_diagonal(Pcovaraince, 0.0)
    CorreScore=np.sum(CovWeight[:,None]*Pcovaraince,axis=0)
    RankLength=min(len(w_k),CandidateLength)
    RankingScore=QPScore+risk_preference_param*CorreScore
    ranklist=np.argsort(-RankingScore)[:RankLength]
    # ranklist=list(ranklist)
    return ranklist
def getNextMaximalOne(score_qid_mean,
                        score_qid_cov,
                        selected_item_id,
                        risk_preference_param,
                        w_k,
                        **param):
    """_summary_
    this funciton selects the next one item when constructing a ranklist.
    Plese refer Eq.9 in the paper https://dl.acm.org/doi/pdf/10.1145/1571941.1571963
    Args:
        score_qid_mean (_type_):  numpy array shape=(NumDoc)
        score_qid_cov (_type_):  numpy array shape=(NumDoc,NumDoc)
        selected_item_id (_type_):   already selected items, list  
        risk_preference_param (_type_):   risk preference parameter, i.e. b in eq.9
        w_k (_type_): _description_ : the weight we put on rank k. e.g., we can use ndcg weight. i.e. w_k in eq.9

    Returns:
        _type_: _description_
    """
    if len(selected_item_id)==0:
        return score_qid_mean.argmax()
    numItems=score_qid_mean.shape[0]
    selected_item_id=np.array(selected_item_id)
    Candidate_id=np.delete(np.arange(numItems),selected_item_id)

    current_rank=len(selected_item_id)
    First=score_qid_mean[Candidate_id] #first part E[r_k] in Eq.9 of https://dl.acm.org/doi/pdf/10.1145/1571941.1571963

    Second=-risk_preference_param*w_k[current_rank]*score_qid_cov[Candidate_id,Candidate_id] #second part  -b*w_k*\sigma_k^2 in Eq.9
    Third=-2*risk_preference_param*np.sum(score_qid_cov[Candidate_id[:,None],selected_item_id[None,:]]*w_k[:current_rank],axis=1) # third part -2b*\sum_{i=1}^{k-1} w_i* cov_{i,k} in Eq.9

    marginal_value_cur=First+Second+Third
    SelectedId=Candidate_id[np.argmax(marginal_value_cur)] # And we select the one with maximal marginal gain. 
    return SelectedId

def EvalRankList(qrel,trecRanklists,metrics=None):
    """_summary_

    Args:
        qrel (_type_): qrel is  qid, “Q0”, docid, rating
        trecRanklists (_type_): qid,Q0,pid,rank,score,desc
        metrics: what metrics we want to evluate.

    Returns:
        _type_: AggResults:aggregated results
                EvalResults: Detailed results of each query.
    """
    assert os.path.exists(qrel)
    assert os.path.exists(trecRanklists)

    # with open(qrel, 'r') as f_qrel:
    qrel = pytrec_eval.parse_qrel(open(qrel, 'r'))

    # with open(run, 'r') as f_run:
    run = pytrec_eval.parse_run(open(trecRanklists, 'r'))
    metrics=pytrec_eval.supported_measures if metrics is None else metrics
    measures=[]
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, pytrec_eval.supported_measures)

    EvalResults = evaluator.evaluate(run)

    # def print_line(measure, scope, value):
    #     print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    # for query_id, query_measures in sorted(results.items()):
    #     for measure, value in sorted(query_measures.items()):
    #         print_line(measure, query_id, value)

    # Scope hack: use query_measures of last item in previous loop to
    # figure out all unique measure names.
    #
    # TODO(cvangysel): add member to RelevanceEvaluator
    #                  with a list of measure names.
    AggResults= defaultdict(list)
    AggResults["iterations"].append(0)
    RealCalMetrics=list(EvalResults.values())[0].keys()
    for measure in sorted(RealCalMetrics):
        AggResults[measure].append(pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure]
                    for query_measures in EvalResults.values()]))
    return AggResults,EvalResults
def BM25FirstPhase(data_path,RanklistPath,QrelsPath,split="test"):
    """_summary_

    Args:
        data_path (_type_): _description_
        outfile (_type_): _description_
    """
    
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split) # or split = "train" or "dev"
    corpus_ids=list(corpus.keys())
    corpuslist = [corpus[cid] for cid in corpus_ids]
    # if type(corpus) is dict:
    #     sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
    # else:
    corpusProcessed = [(doc["title"] + " " + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpuslist]
    tokenized_corpus = [doc.split(" ") for doc in corpusProcessed]
    bm25 = BM25Okapi(tokenized_corpus)
    FirstPhaseLength=1000
    # BM25RanklistFile=os.path.join(outPath,"BM25Top1000.tsv")
    with open(RanklistPath, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        
        for queryId in tqdm(queries, desc ="Generating ranklists",total=len(queries.keys())):
            query=queries[queryId]
            tokenized_query = query.split(" ")

            # doc_scores = bm25.get_top_n(tokenized_query,corpus,n=1)
            # print(doc_scores)
            doc_scores = bm25.get_scores(tokenized_query)
            TopKind = np.argpartition(-doc_scores, FirstPhaseLength)[:FirstPhaseLength]
            for LocalPid in TopKind:
                tsv_writer.writerow([queryId,corpus_ids[LocalPid],queries[queryId],corpus[corpus_ids[LocalPid]]])  ## output to trec format
    with open(QrelsPath, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for qid in qrels:
            for pid in qrels[qid]:
                tsv_writer.writerow([qid,"Q0",pid,qrels[qid][pid]])  ## output to trec format