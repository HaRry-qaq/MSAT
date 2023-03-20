import argparse
import logging
import pathlib
import pprint
import sys
from collections import defaultdict
import pandas as pd
import muspy
import numpy as np
import torch
import torch.utils.data
import tqdm
sys.path.append(r'/work100/weixp/MSMM-main/baseline/MTMT-bar')
import representation
import utils
from csv import reader
import os
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def csv_read_instruments(inputPath):
    csv_result = pd.read_csv(inputPath)
    row_list = csv_result.values.tolist()
    count =0
    # print(row_list)
    instruments =[]
    for r in row_list:
          if r[0]==3 and r[5] not in instruments:
                count+=1
                instruments.append(r[5])
    return count



def get_all_instruments(data):
    instruments =[]
    for r in data:
        if r[0]==3 and r[5] not in instruments:
                instruments.append(r[5]);
    
    return instruments



def pre_process(file):
    with open(file, 'r',encoding='latin-1') as f:
        data = list(reader(f))
    data = np.array(data[1:],dtype='int')
    
    instruments =[]
    for r in data:
        if r[1]<=1024:
          if r[0]==3 and r[5] not in instruments:
                instruments.append(r[5])
    
    data_final=[]
    for r in data:
        if r[0]==1 and r[5] in instruments:
            data_final.append(r)
        if  r[0]!=1 and r[0]!=3:
            data_final.append(r)  
        if r[0]==3 and r[1]<=1024:
            data_final.append(r)
    return data_final


def main():
    # InputPath = "/work100/weixp/mtmt3-paper-2/mtmt-baseline-220/exp/lmd-track/ape/samples/csv"
    
    InputPath = "/work104/weixp/mtmt3-paper/mtmt3-final-global_attetion/exp/lmd-bar/ape/samples/csv"
    
    InputPath_file=pathlib.Path(InputPath)
    encoding = representation.load_encoding("./encoding.json")
    # Parse the command-line arguments
    args = parse_args()
    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(InputPath_file / "evaluate.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    results_truth = []
    results_out =[]

    InputPath_truth = InputPath+'/'+"truth"
    datanames1 = os.listdir(InputPath_truth)
    for i1 in datanames1:
        if i1[-3:]=="csv":
            # print(i1)
            path1 = InputPath_truth+'/'+i1
            data_csv0 = pre_process(path1)
            instru_number1 = get_all_instruments(data_csv0)
            if len(instru_number1)<=1:
                continue
            
            results_truth.append(len(instru_number1))
            
            path2 = InputPath+'/instrument-beats'+'/'+i1.split('_')[0]+'_instrument-continuation.csv'
            
            # ------------------------
            # Unconditioned generation
            # ------------------------
            # Evaluate the results
            data_csv2 = pre_process(path2)
            instru_number2 = get_all_instruments(data_csv2)
 
            results_out.append(len(instru_number2))
    Mat1 = np.array(results_out)
    Mat2 = np.array(results_truth)
    # print('矩阵')
    # print(Mat1)
    # print(Mat2)
    # print(results_out)
    print('相关系数为：')
    corr = np.corrcoef(Mat1,Mat2)         
    print(corr)

    print(len(results_out))
    print(len(results_truth))


if __name__ == "__main__":
    main()