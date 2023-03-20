import csv
import pandas as pd
import os


def csv_read(inputPath,pathout):
    csv_result = pd.read_csv(inputPath)
    row_list = csv_result.values.tolist()
    
    row_list.sort(key=lambda x: (x[0]//4,x[4],x[0],x[1],x[2],x[3],x[4]))
    
    track_all=[]
    for r in row_list:
            track_all.append(r);                        
    header =['beat','position','pitch','duration','program']
    test=pd.DataFrame( columns=header,data=track_all)
    test.to_csv(pathout, encoding= 'gbk',index=False)
    
if __name__ == '__main__':
  
      InputPath = "/work100/weixp/MSMM-main-final1-test/baseline/data/sod/processed/notes-1024/Kunstderfuge"
      OutputPath = '/work100/weixp/MSMM-main-final1-test/baseline/data/sod/processed/notes-bar/Kunstderfuge'
      datanames = os.listdir(InputPath)
      for i in datanames:
              # print(i)
            if i=='encoding.json':
                continue
            if i[-3:]=='csv':
                path1 = InputPath+'/'+i
                pathout1 = OutputPath+'/'+i
                csv_read(path1,pathout1)
                print(path1+" save to :"+pathout1)
  
  
  