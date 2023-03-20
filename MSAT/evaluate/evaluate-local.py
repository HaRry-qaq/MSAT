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

def evaluate_track(data,encoding,tracknumber):

    track_0=[]
    for r in data:
        if r[1]<=1024:
            if r[0]==1 and r[5]==tracknumber:
                track_0.append(r);
            if r[0]!=3 and r[0]!=1:
                track_0.append(r);
            if r[0]==3 and r[5]==tracknumber:
                track_0.append(r);
    music1 = representation.decode(track_0, encoding)

    if not music1.tracks:
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
        }
    return {
        "pitch_class_entropy": muspy.pitch_class_entropy(music1),
        "scale_consistency": muspy.scale_consistency(music1),
        "groove_consistency": muspy.groove_consistency(
            music1, 4 * music1.resolution
        ),  
    }
    
    
def evaluate(data, encoding, filepath):
    """Evaluate the results."""
    path1 =filepath
    flag= True
    instruments =[]
    flag = True
    for r in data:
        if r[0]==3 and r[1]<=1024 and r[5] not in instruments:
                instruments.append(r[5]);
    if  len(instruments)==0:
            flag = False
    else:
        instruments_number0=len(instruments)
        
    
    
    set_all={
        "pitch_class_entropy": 0,
        "scale_consistency": 0,
        "groove_consistency": 0,
        
    }

    for i in instruments:
        set_this=evaluate_track(data,encoding,i)
        set_all['pitch_class_entropy']+=set_this['pitch_class_entropy']
        set_all['scale_consistency']+=set_this['scale_consistency']
        set_all['groove_consistency']+=set_this['groove_consistency']
        
    
    if len(instruments)==0:
        set_all['pitch_class_entropy']=0
        set_all['scale_consistency']=0
        set_all['groove_consistency']=0
        
    else:
        set_all['pitch_class_entropy']=set_all['pitch_class_entropy']/len(instruments)
        set_all['scale_consistency']=set_all['scale_consistency']/len(instruments)
        set_all['groove_consistency']=set_all['groove_consistency']/len(instruments)
       
   
    
    return set_all

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
def get_all_instruments(data):
    instruments =[]
    for r in data:
        if r[0]==3 and r[5] not in instruments:
                instruments.append(r[5]);
    
    return instruments




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
    
    results = defaultdict(list)
    truth_number= 0
    InputPath_truth = InputPath+'/'+"truth"
    datanames1 = os.listdir(InputPath_truth)
    for i1 in datanames1:
        if i1[-3:]=="csv":
            # print(i1)
            path1 = InputPath_truth+'/'+i1
            data_csv0 = pre_process(path1)
            instru_number0 = get_all_instruments(data_csv0)
            if len(instru_number0)<=1:
                continue
            
            
            result = evaluate(
                data_csv0, encoding,path1
            )
            results["truth"].append(result)
            truth_number+=1
    
    unconditioned_number= 0
    InputPath_unconditioned = InputPath+'/instrument-beats'
    datanames2 = os.listdir(InputPath_unconditioned)
    
  
    for i2 in datanames2:
        if i2[-3:]=="csv":
            # print(i2)

            path2 = InputPath_unconditioned+'/'+i2
            
            # ------------------------
            # Unconditioned generation
            # ------------------------
            # Evaluate the results
            data_csv2 = pre_process(path2)
            
            
            result = evaluate(
                data_csv2,
                encoding,
                path2
            )
            results["instrument"].append(result)
            unconditioned_number+=1

    for exp, result in results.items():
        logging.info(exp)
        for key in result[0]:
            logging.info(
                f"{key}: mean={np.nanmean([r[key] for r in result]):.4f}, "
                f"steddev={np.nanstd([r[key]for r in result]):.4f}"
            )
    print("truth")
    print(truth_number)
    print("instrument")
    print(unconditioned_number)
    print(InputPath)

if __name__ == "__main__":
    main()
