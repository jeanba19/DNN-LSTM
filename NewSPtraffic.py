#!/usr/bin/env python
# coding: utf-8




from InPT_trace_preparation import *
num_slots_day=144
source_file_length_days=31
m=15/100
n_time_slots=num_slots_day*source_file_length_days





import numpy as np
import numpy.random as nr


class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


ou = OUNoise(1)
states = []
for i in range(source_file_length_days*num_slots_day):
    states.append(ou.noise())





SPT_h1=[]
SPT_h2=[]
SPT_h3=[]
SPT_h4=[]
SPT_h5=[]
SPT_h6=[]
SPT_h7=[]
SPT_h8=[]
SPT_h9=[]
SPT_h10=[]
SPT_w1=[]
SPT_w2=[]
SPT_w3=[]
SPT_w4=[]
SPT_w5=[]
SPT_w6=[]
SPT_w7=[]
SPT_w8=[]
SPT_w9=[]
SPT_w10=[]





for t in range(source_file_length_days*num_slots_day):
    SPT_h1.append( max( m * h1_InPT_trace[t] + states[t] , 0 ) )
    SPT_h2.append( max( m * h2_InPT_trace[t] + states[t] , 0 ) )
    SPT_h3.append( max( m * h3_InPT_trace[t] + states[t] , 0 ) )
    SPT_h4.append( max( m * h4_InPT_trace[t] + states[t] , 0 ) )
    SPT_h5.append( max( m * h5_InPT_trace[t] + states[t] , 0 ) )
    SPT_h6.append( max( m * h6_InPT_trace[t] + states[t] , 0 ) )
    SPT_h7.append( max( m * h7_InPT_trace[t] + states[t] , 0 ) )
    SPT_h8.append( max( m * h8_InPT_trace[t] + states[t] , 0 ) )
    SPT_h9.append( max( m * h9_InPT_trace[t] + states[t] , 0 ) )
    SPT_h10.append( max( m * h10_InPT_trace[t] + states[t] , 0 ) )
    SPT_w1.append( max( m * w1_InPT_trace[t] + states[t] , 0 ) )
    SPT_w2.append( max( m * w2_InPT_trace[t] + states[t] , 0 ) )
    SPT_w3.append( max( m * w3_InPT_trace[t] + states[t] , 0 ) )
    SPT_w4.append( max( m * w4_InPT_trace[t] + states[t] , 0 ) )
    SPT_w5.append( max( m * w5_InPT_trace[t] + states[t] , 0 ) )
    SPT_w6.append( max( m * w6_InPT_trace[t] + states[t] , 0 ) )
    SPT_w7.append( max( m * w7_InPT_trace[t] + states[t] , 0 ) )
    SPT_w8.append( max( m * w8_InPT_trace[t] + states[t] , 0 ) )
    SPT_w9.append( max( m * w9_InPT_trace[t] + states[t] , 0 ) )
    SPT_w10.append( max( m * w10_InPT_trace[t] + states[t] , 0 ) )



perc0 = sum(i==0 for i in np.array(SPT_h8).astype(float))/len(SPT_h1)
perc0
#print("the percentage of zeros is {:2f} %".format(perc0*100))



x=range(4464)
y=SPT_h1
import matplotlib.pyplot as plt
plt.plot(x,y)




df=pd.DataFrame()
df['SP']=SPT_h1[:400]
df['MNO']=h1_InPT_trace[:400]
df['CAP']=np.percentile(h1_InPT_trace, 80)
plt.figure(figsize=(16,8))
plt.plot(df['MNO'], label="MNO traffic")
plt.plot(df['SP'], label="SP traffic")
plt.plot(df['CAP'], label="capacity")
plt.legend(loc='best')




SPT_h1=np.array(SPT_h1).astype(float)
SPT_h2=np.array(SPT_h2).astype(float)
SPT_h3=np.array(SPT_h3).astype(float)
SPT_h4=np.array(SPT_h4).astype(float)
SPT_h5=np.array(SPT_h5).astype(float)
SPT_h6=np.array(SPT_h6).astype(float)
SPT_h7=np.array(SPT_h7).astype(float)
SPT_h8=np.array(SPT_h8).astype(float)
SPT_h9=np.array(SPT_h9).astype(float)
SPT_h10=np.array(SPT_h10).astype(float)
SPT_w1=np.array(SPT_w1).astype(float)
SPT_w2=np.array(SPT_w2).astype(float)
SPT_w3=np.array(SPT_w3).astype(float)
SPT_w4=np.array(SPT_w4).astype(float)
SPT_w5=np.array(SPT_w5).astype(float)
SPT_w6=np.array(SPT_w6).astype(float)
SPT_w7=np.array(SPT_w7).astype(float)
SPT_w8=np.array(SPT_w8).astype(float)
SPT_w9=np.array(SPT_w9).astype(float)
SPT_w10=np.array(SPT_w10).astype(float)



SPT_h10




