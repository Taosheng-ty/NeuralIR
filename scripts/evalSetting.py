import sys
import json
import os
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../")
import BatchExpLaunch.tools as tools

import os
import json
import os
rootpath="output/mse-huggingfaceHard10EpochDist/"
rootpath="output/mse-huggingfaceHard10EpochCo-con/"
rootpath="output/SentenceBertOutput/MSEOutputNoPreTrained/"
subdir=os.listdir(path=rootpath)
for path in subdir:
    if os.path.isdir(rootpath+path):
        cmd="slurmSingle --Cmd_file=eval.py --OutputDir={rootpath}/{path}/Eval --Cmd_args='--model_name={rootpath}/{path} --log_dir={rootpath}/{path}/Eval'".format(rootpath = rootpath, path = path)
        print(cmd)
        os.system(cmd)