#data processing
import pandas as pd
data = pd.read_csv("/home/jeanba19/Project/Dataset190719/traffic.txt", header=None, names=["BaseID", "Coordinates", "TrafficVector"], sep="\t")

for i in range(data.shape[0]):
    data["TrafficVector"][i] = data["TrafficVector"][i].split('|')
    del data["TrafficVector"][i][-1]
    data["TrafficVector"][i] = [int(j) for j in data["TrafficVector"][i]]
    
import numpy as np
l_avg=[]
for x in data["TrafficVector"]:
    avg = np.mean(x)
    l_avg.append(avg)
data["VectorAverage"]=l_avg
norm_const = np.mean(data["VectorAverage"])
data["VectorAverage"]=data["VectorAverage"]/norm_const

#Normalization step, we divide each element by norm_const
L=[]
for x in data["TrafficVector"]:
    L.append(x)
import numpy as np
M=np.array(L)
M = M/norm_const

#Generation of the trace at each eNb
import pandas as pd
Vectors=pd.DataFrame(M)
data = pd.concat([data, Vectors], axis=1)

data = data.drop(["TrafficVector"], axis=1)
data = data.drop(["Coordinates"], axis=1)
data = data.drop(["VectorAverage"], axis=1)
h1_InPT_trace = data.iloc[0][1:]
h2_InPT_trace = data.iloc[1][1:]
h3_InPT_trace = data.iloc[2][1:]
h4_InPT_trace = data.iloc[3][1:]
h5_InPT_trace = data.iloc[4][1:]
h6_InPT_trace = data.iloc[5][1:]
h7_InPT_trace = data.iloc[6][1:]
h8_InPT_trace = data.iloc[7][1:]
h9_InPT_trace = data.iloc[8][1:]
h10_InPT_trace = data.iloc[9][1:]
w1_InPT_trace = data.iloc[10][1:]   
w2_InPT_trace = data.iloc[11][1:]
w3_InPT_trace = data.iloc[12][1:]
w4_InPT_trace = data.iloc[13][1:]
w5_InPT_trace = data.iloc[14][1:]
w6_InPT_trace = data.iloc[15][1:]
w7_InPT_trace = data.iloc[16][1:]
w8_InPT_trace = data.iloc[17][1:]
w9_InPT_trace = data.iloc[18][1:]
w10_InPT_trace = data.iloc[19][1:]


