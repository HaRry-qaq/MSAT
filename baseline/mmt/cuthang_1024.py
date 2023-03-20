import pandas as pd
import os




# 1021
def csv_read(inputPath,pathout):
    csv_result = pd.read_csv(inputPath)
    row_list = csv_result.values.tolist()
    
    track_all = []
    instruments =[]
    count_r= 0
    for r in row_list:
        if count_r + len(instruments) >=1021:
            break
        if r[4] not in instruments:
            instruments.append(r[4]);
        track_all.append(r);
        count_r+=1
    
                
          
    header =['beat','position','pitch','duration','program']
    test=pd.DataFrame( columns=header,data=track_all)
# 转换csv
    print(test)
    test.to_csv(pathout, encoding= 'gbk',index=False)
    
    
    
InputPath = "/work100/weixp/MSMM-main-final1-test/baseline/data/sod/processed/notes"
outputPath = "/work100/weixp/MSMM-main-final1-test/baseline/data/sod/processed/notes-1024"
datanames = os.listdir(InputPath)

for i in datanames:
    if i=='encoding.json':
          continue
    path = InputPath+'/'+i
    pathout = outputPath+'/'+i
    datanames1 = os.listdir(path)
    for i1 in datanames1:
        if i1[-3:]=="csv":
            print(i1)
            path1 = path+'/'+i1
            pathout1 = pathout+'/'+i1
            csv_read(path1,pathout1)
            
