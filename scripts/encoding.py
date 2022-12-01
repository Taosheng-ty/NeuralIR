import sys
import json
import os
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
import BatchExpLaunch.tools as tools

list_settings={'Model':["msmarco-distilbert-base-tas-b","msmarco-distilbert-dot-v5","msmarco-bert-base-dot-v5"]}
base_settings={}
root_path="output/ModelEncoding/"
## The following funciton will iteratively expand list_settings to create  {"DataSet":"DL19",'Model':"msmarco-distilbert-base-tas-b","log_dir":"output/DataSet_DL19/Model_msmarco-distilbert-base-tas-b"}
tools.iterate_settings(list_settings,base_settings,path=root_path)