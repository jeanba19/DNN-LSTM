#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, CuDNNLSTM
from sklearn import preprocessing
import random
import h5py
from keras.optimizers import Adam
import matplotlib.pyplot as plt


net_type_num=1
numTrainingEpochs=10
scramble_data=True
num_of_repts=10
traces=[6,18,36,60]
percentiles=[70,80,90]




#input dimensions
from InPT_trace_preparation import *
from NewSPtraffic import *
net_type=['DNN','RNN']
numFeatures=4
num_of_repts=10
testSplit=0.1




def splitTrainTest(datalength, testSplit):
    num_train_samples = int(datalength*(1-testSplit))
    num_test_samples = datalength - num_train_samples
    return num_train_samples, num_test_samples




def labelForMultipleStepsPrediction(label_vector, nobs=9):
    u2 = np.roll(label_vector, -1)
    u3 = np.roll(label_vector, -2)
    u4 = np.roll(label_vector, -3)
    u5 = np.roll(label_vector, -4)
    u6 = np.roll(label_vector, -5)
    u7 = np.roll(label_vector, -6)
    u8 = np.roll(label_vector, -7)
    u9 = np.roll(label_vector, -8)
    return np.concatenate((label_vector[:-nobs+1],u2[1:-nobs+2],u3[2:-nobs+3],u4[3:-nobs+4],u5[4:-nobs+5],u6[5:-nobs+6],u7[6:-nobs+7],u8[7:-nobs+8],u9[8:]), axis=1)



#generate traffic traces for reserved traffic, best effort traffic, unserved traffic, load and served traffic
from random import *

flatten = lambda l: [item for sublist in l for item in sublist]

def generate_data(InPT_trace, SPT, tracelength=36, cap_value=80):
    
    D= np.zeros((n_time_slots))
    U= np.zeros((n_time_slots))
    B= np.zeros((n_time_slots))
    Ropt= np.zeros((n_time_slots))
    
    R=np.random.rand(source_file_length_days*num_slots_day)*max(SPT)
    cap=np.percentile(InPT_trace, cap_value)
    
    for k in range(n_time_slots):
        if SPT[k] + InPT_trace[k] <= cap:
            D[k] = SPT[k]
            U[k] = 0
            B[k] = max(D[k]-R[k],0)
            
            Ropt[k] = 0
        else:
            D[k] = min(max(cap-InPT_trace[k]+R[k],R[k]),SPT[k])
            U[k] = SPT[k] - D[k]
            B[k] = max(D[k]-R[k], 0)
            
            Ropt[k] = min(min(SPT[k]-(cap-InPT_trace[k]),SPT[k]),cap)
      
    x = [flatten([SPT[ii:(ii+tracelength)], U[ii:(ii+tracelength)], B[ii:(ii+tracelength)], R[ii:(ii+tracelength)]]) for ii in range(n_time_slots-tracelength-nobs+1)]
    x = np.array(x)
    
    y=np.array(Ropt[tracelength:]).reshape(4464-tracelength,1)
    y=labelForMultipleStepsPrediction(y)
    
    
    return x, y




nobs=9




trainingExamples={}
trainingLabels={}

testExamples={}
testLabels={}
for x in traces:
    for y in percentiles:
        
        X_h1 = generate_data(h1_InPT_trace, SPT_h1, tracelength=x, cap_value=y)
        X_h2 = generate_data(h2_InPT_trace, SPT_h2, tracelength=x, cap_value=y)
        X_h3 = generate_data(h3_InPT_trace, SPT_h3, tracelength=x, cap_value=y)
        X_h4 = generate_data(h4_InPT_trace, SPT_h4, tracelength=x, cap_value=y)
        X_h5 = generate_data(h5_InPT_trace, SPT_h5, tracelength=x, cap_value=y)
        X_h6 = generate_data(h6_InPT_trace, SPT_h6, tracelength=x, cap_value=y)
        X_h7 = generate_data(h7_InPT_trace, SPT_h7, tracelength=x, cap_value=y)
        X_h8 = generate_data(h8_InPT_trace, SPT_h8, tracelength=x, cap_value=y)
        X_h9 = generate_data(h9_InPT_trace, SPT_h9, tracelength=x, cap_value=y)
        X_h10 = generate_data(h10_InPT_trace, SPT_h10, tracelength=x, cap_value=y)
        X_w2 = generate_data(w2_InPT_trace, SPT_w2, tracelength=x, cap_value=y)
        X_w3 = generate_data(w3_InPT_trace, SPT_w3, tracelength=x, cap_value=y)
        X_w6 = generate_data(w6_InPT_trace, SPT_w6, tracelength=x, cap_value=y)
        X_w9 = generate_data(w9_InPT_trace, SPT_w9, tracelength=x, cap_value=y)

        h1_train = X_h1[0][:splitTrainTest(len(X_h1[0]), testSplit)[0],:]
        h2_train = X_h2[0][:splitTrainTest(len(X_h2[0]), testSplit)[0],:]
        h3_train = X_h3[0][:splitTrainTest(len(X_h3[0]), testSplit)[0],:]
        h4_train = X_h4[0][:splitTrainTest(len(X_h4[0]), testSplit)[0],:]
        h5_train = X_h5[0][:splitTrainTest(len(X_h5[0]), testSplit)[0],:]
        h6_train = X_h6[0][:splitTrainTest(len(X_h6[0]), testSplit)[0],:]
        h7_train = X_h7[0][:splitTrainTest(len(X_h7[0]), testSplit)[0],:]
        h8_train = X_h8[0][:splitTrainTest(len(X_h8[0]), testSplit)[0],:]
        h9_train = X_h9[0][:splitTrainTest(len(X_h9[0]), testSplit)[0],:]
        h10_train = X_h10[0][:splitTrainTest(len(X_h10[0]), testSplit)[0],:]
        w2_train = X_w2[0][:splitTrainTest(len(X_w2[0]), testSplit)[0],:]
        w3_train = X_w3[0][:splitTrainTest(len(X_w3[0]), testSplit)[0],:]
        w6_train = X_w6[0][:splitTrainTest(len(X_w6[0]), testSplit)[0],:]
        w9_train = X_w9[0][:splitTrainTest(len(X_w9[0]), testSplit)[0],:]

        h1_label = X_h1[1][:splitTrainTest(len(X_h1[1]), testSplit)[0],:]
        h2_label = X_h2[1][:splitTrainTest(len(X_h2[1]), testSplit)[0],:]
        h3_label = X_h3[1][:splitTrainTest(len(X_h3[1]), testSplit)[0],:]
        h4_label = X_h4[1][:splitTrainTest(len(X_h4[1]), testSplit)[0],:]
        h5_label = X_h5[1][:splitTrainTest(len(X_h5[1]), testSplit)[0],:]
        h6_label = X_h6[1][:splitTrainTest(len(X_h6[1]), testSplit)[0],:]
        h7_label = X_h7[1][:splitTrainTest(len(X_h7[1]), testSplit)[0],:]
        h8_label = X_h8[1][:splitTrainTest(len(X_h8[1]), testSplit)[0],:]
        h9_label = X_h9[1][:splitTrainTest(len(X_h9[1]), testSplit)[0],:]
        h10_label = X_h10[1][:splitTrainTest(len(X_h10[1]), testSplit)[0],:]
        w2_label = X_w2[1][:splitTrainTest(len(X_w2[1]), testSplit)[0],:]
        w3_label = X_w3[1][:splitTrainTest(len(X_w3[1]), testSplit)[0],:]
        w6_label = X_w6[1][:splitTrainTest(len(X_w6[1]), testSplit)[0],:]
        w9_label = X_w9[1][:splitTrainTest(len(X_w9[1]), testSplit)[0],:]

        h1_test = X_h1[0][-splitTrainTest(len(X_h1[0]), testSplit)[1]:,:]
        h2_test = X_h2[0][-splitTrainTest(len(X_h2[0]), testSplit)[1]:,:]
        h3_test = X_h3[0][-splitTrainTest(len(X_h3[0]), testSplit)[1]:,:]
        h4_test = X_h4[0][-splitTrainTest(len(X_h4[0]), testSplit)[1]:,:]
        h5_test = X_h5[0][-splitTrainTest(len(X_h5[0]), testSplit)[1]:,:]
        h6_test = X_h6[0][-splitTrainTest(len(X_h6[0]), testSplit)[1]:,:]
        h7_test = X_h7[0][-splitTrainTest(len(X_h7[0]), testSplit)[1]:,:]
        h8_test = X_h8[0][-splitTrainTest(len(X_h8[0]), testSplit)[1]:,:]
        h9_test = X_h9[0][-splitTrainTest(len(X_h9[0]), testSplit)[1]:,:]
        h10_test = X_h10[0][-splitTrainTest(len(X_h10[0]), testSplit)[1]:,:]
        w2_test = X_w2[0][-splitTrainTest(len(X_w2[0]), testSplit)[1]:,:]
        w3_test = X_w3[0][-splitTrainTest(len(X_w3[0]), testSplit)[1]:,:]
        w6_test = X_w6[0][-splitTrainTest(len(X_w6[0]), testSplit)[1]:,:]
        w9_test = X_w9[0][-splitTrainTest(len(X_w9[0]), testSplit)[1]:,:]

        h1_label2 = X_h1[1][-splitTrainTest(len(X_h1[0]), testSplit)[1]:,:]
        h2_label2 = X_h2[1][-splitTrainTest(len(X_h2[0]), testSplit)[1]:,:]
        h3_label2 = X_h3[1][-splitTrainTest(len(X_h3[0]), testSplit)[1]:,:]
        h4_label2 = X_h4[1][-splitTrainTest(len(X_h4[0]), testSplit)[1]:,:]
        h5_label2 = X_h5[1][-splitTrainTest(len(X_h5[0]), testSplit)[1]:,:]
        h6_label2 = X_h6[1][-splitTrainTest(len(X_h6[0]), testSplit)[1]:,:]
        h7_label2 = X_h7[1][-splitTrainTest(len(X_h7[0]), testSplit)[1]:,:]
        h8_label2 = X_h8[1][-splitTrainTest(len(X_h8[0]), testSplit)[1]:,:]
        h9_label2 = X_h9[1][-splitTrainTest(len(X_h9[0]), testSplit)[1]:,:]
        h10_label2 = X_h10[1][-splitTrainTest(len(X_h10[0]), testSplit)[1]:,:]
        w2_label2 = X_w2[1][-splitTrainTest(len(X_w2[0]), testSplit)[1]:,:]
        w3_label2 = X_w3[1][-splitTrainTest(len(X_w3[0]), testSplit)[1]:,:]
        w6_label2 = X_w6[1][-splitTrainTest(len(X_w6[0]), testSplit)[1]:,:]
        w9_label2 = X_w9[1][-splitTrainTest(len(X_w9[0]), testSplit)[1]:,:]



        trainingExamples.update({ (x,y) : np.concatenate((h1_train, h2_train, h3_train, h4_train, h5_train, h6_train, h7_train, h8_train, h9_train, h10_train, w2_train, w3_train, w6_train, w9_train)) })
        trainingLabels.update({ (x,y) : np.concatenate((h1_label, h2_label, h3_label, h4_label, h5_label, h6_label, h7_label, h8_label, h9_label, h10_label, w2_label, w3_label, w6_label, w9_label))})
        testExamples.update({ (x,y) : np.concatenate((h1_test, h2_test, h3_test, h4_test, h5_test, h6_test, h7_test, h8_test, h9_test, h10_test, w2_test, w3_test, w6_test, w9_test))})
        testLabels.update({ (x,y) : np.concatenate((h1_label2, h2_label2, h3_label2, h4_label2, h5_label2, h6_label2, h7_label2, h8_label2, h9_label2, h10_label2, w2_label2, w3_label2, w6_label2, w9_label2))})


        
        
        

testExamples[(36,70)].shape



trainingExamples[(36,70)].shape


scenarios=trainingExamples.keys()



#duplicate attempt fo train and test
num_of_repts=10

#repts_for_test=1


for key, item in trainingExamples.items():
    trainingExamples[key] = np.tile(item, (num_of_repts,1))
    
for key, item in trainingLabels.items():
    trainingLabels[key] = np.tile(item, (num_of_repts,1))

    

#for key, item in testExamples.items():
#    testExamples[key] = np.tile(item, (repts_for_test,1))
    
#for key, item in testLabels.items():
#    testLabels[key] = np.tile(item, (repts_for_test,1))




train_permutation=dict.fromkeys(scenarios)
test_permutation=dict.fromkeys(scenarios)

if scramble_data:
    print("Scramble data activated")
    for key, item in trainingExamples.items():
        train_permutation[key] = np.random.permutation(range(0, item.shape[0]))
        item = item[train_permutation[key][range(0, item.shape[0])], :]
    for key, item in trainingLabels.items():
        item = item[train_permutation[key][range(0, item.shape[0])], :]

else:
    print("No scrambling, keep input data unchanged") 




#training the DNN/RNN

numTrainingEpochs=10

#model_DNN = dict.fromkeys(scenarios)
i = 0
for (k,v), (k2,v2) in zip(trainingExamples.items(), trainingLabels.items()):

    model_DNN = Sequential()

    model_DNN.add(Dense(50, activation='tanh', input_dim = v.shape[1]))
    model_DNN.add(Dense(10, activation='tanh'))
    model_DNN.add(Dense(nobs, activation='sigmoid'))
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
    model_DNN.compile(loss = 'mse', optimizer = 'Adam', metrics = ["accuracy"])
    #config = model_DNN[key].summary()
    history = model_DNN.fit(v, v2, validation_split= 0.1, epochs=numTrainingEpochs)
    #config = model_DNN[key].summary()
    model_DNN.save_weights("DNN_weights/file_DNN_weights_{}.h5".format(k))
    i = i + 1

    











