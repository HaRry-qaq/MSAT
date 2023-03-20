import csv
import pandas as pd
import os
def csv_read(inputPath,pathout):
    csv_result = pd.read_csv(inputPath)
    row_list = csv_result.values.tolist()
    instruments =[]
    for r in row_list:
          if r[4] not in instruments:
                instruments.append(r[4]);
    track_all = []
    
    for instru in instruments:
          track_this = []
          for r in row_list:
                if r[4]==instru:
                      track_this.append(r)
          track_all=track_all+track_this
                    
                
          
    header =['beat','position','pitch','duration','program']
    test=pd.DataFrame( columns=header,data=track_all)
    print(test)
    test.to_csv(pathout, encoding= 'gbk',index=False)
    
if __name__ == '__main__':
  
      InputPath = "/work100/weixp/MSMM-main-final1-test/baseline/data/sod/processed/notes-1024/Kunstderfuge"
      OutputPath = '/work100/weixp/MSMM-main-final1-test/baseline/data/sod/processed/notes-track/Kunstderfuge'
      datanames = os.listdir(InputPath)
      for i in datanames:
            if i=='encoding.json':
                  continue
            if i[-3:]=='csv':
                  path1 = InputPath+'/'+i
                  pathout1 = OutputPath+'/'+i
                  csv_read(path1,pathout1)
                  print(path1+" save to :"+pathout1)
     
      
  
  
  