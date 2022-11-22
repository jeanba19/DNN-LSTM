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



#training parameters
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

trace_length=36
num_of_repts=10
testSplit=0.1




nobs=9



def splitTrainTest(datalength, testSplit):
    num_train_samples = int(datalength*(1-testSplit))
    num_test_samples = datalength - num_train_samples
    return num_train_samples, num_test_samples


#generate traffic traces for reserved traffic, best effort traffic, unserved traffic, load and served traffic
from random import *

flatten = lambda l: [item for sublist in l for item in sublist]

def generate_data(InPT_trace, SPT, seq_length=36, cap_value=80):
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
      #concatenate column vectors
    x=np.concatenate((SPT.reshape(4464,1),U.reshape(4464,1),B.reshape(4464,1),R.reshape(4464,1)), axis=1)
    
    y=np.array(Ropt).reshape(4464,1)
    y=labelForMultipleStepsPrediction(y)
    
    m = x.shape[0]

    x1 = np.zeros([m - seq_length - nobs+1, seq_length, x.shape[1]])
    y1 = np.zeros([m - seq_length - nobs+1, y.shape[1]]) # output_len = 1

    for i in range(x1.shape[0]):
        x1[i, :, :] = x[i:i + seq_length]
        y1[i, :] = y[i + seq_length]
    return x1, y1
    



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


trainingExamples={}
trainingLabels={}
testExamples={}
testLabels={}
for x in traces:
    for y in percentiles:
        h1 = generate_data(h1_InPT_trace, SPT_h1, x, y)
        h2 = generate_data(h2_InPT_trace, SPT_h2, x, y)
        h3 = generate_data(h3_InPT_trace, SPT_h3, x, y)
        h4 = generate_data(h4_InPT_trace, SPT_h4, x, y)
        h5 = generate_data(h5_InPT_trace, SPT_h5, x, y)
        h6 = generate_data(h6_InPT_trace, SPT_h6, x, y)
        h7 = generate_data(h7_InPT_trace, SPT_h7, x, y)
        h8 = generate_data(h8_InPT_trace, SPT_h8, x, y)
        h9 = generate_data(h9_InPT_trace, SPT_h9, x, y)
        h10 = generate_data(h10_InPT_trace, SPT_h10, x, y)
        w2 = generate_data(w2_InPT_trace, SPT_w2, x, y)
        w3 = generate_data(w3_InPT_trace, SPT_w3, x, y)
        w6 = generate_data(w6_InPT_trace, SPT_w6, x, y)
        w9 = generate_data(w9_InPT_trace, SPT_w9, x, y)
        
        h1_train = h1[0][:splitTrainTest(len(h1[0]), testSplit)[0],:,:]
        h2_train = h2[0][:splitTrainTest(len(h2[0]), testSplit)[0],:,:]
        h3_train = h3[0][:splitTrainTest(len(h3[0]), testSplit)[0],:,:]
        h4_train = h4[0][:splitTrainTest(len(h4[0]), testSplit)[0],:,:]
        h5_train = h5[0][:splitTrainTest(len(h5[0]), testSplit)[0],:,:]
        h6_train = h6[0][:splitTrainTest(len(h6[0]), testSplit)[0],:,:]
        h7_train = h7[0][:splitTrainTest(len(h7[0]), testSplit)[0],:,:]
        h8_train = h8[0][:splitTrainTest(len(h8[0]), testSplit)[0],:,:]
        h9_train = h9[0][:splitTrainTest(len(h9[0]), testSplit)[0],:,:]
        h10_train = h10[0][:splitTrainTest(len(h10[0]), testSplit)[0],:,:]
        w2_train = w2[0][:splitTrainTest(len(w2[0]), testSplit)[0],:,:]
        w3_train = w3[0][:splitTrainTest(len(w3[0]), testSplit)[0],:,:]
        w6_train = w6[0][:splitTrainTest(len(w6[0]), testSplit)[0],:,:]
        w9_train = w9[0][:splitTrainTest(len(w9[0]), testSplit)[0],:,:]
        
        h1_label = h1[1][:splitTrainTest(len(h1[1]), testSplit)[0],:]
        h2_label = h2[1][:splitTrainTest(len(h2[1]), testSplit)[0],:]
        h3_label = h3[1][:splitTrainTest(len(h3[1]), testSplit)[0],:]
        h4_label = h4[1][:splitTrainTest(len(h4[1]), testSplit)[0],:]
        h5_label = h5[1][:splitTrainTest(len(h5[1]), testSplit)[0],:]
        h6_label = h6[1][:splitTrainTest(len(h6[1]), testSplit)[0],:]
        h7_label = h7[1][:splitTrainTest(len(h7[1]), testSplit)[0],:]
        h8_label = h8[1][:splitTrainTest(len(h8[1]), testSplit)[0],:]
        h9_label = h9[1][:splitTrainTest(len(h9[1]), testSplit)[0],:]
        h10_label = h10[1][:splitTrainTest(len(h10[1]), testSplit)[0],:]
        w2_label = w2[1][:splitTrainTest(len(w2[1]), testSplit)[0],:]
        w3_label = w3[1][:splitTrainTest(len(w3[1]), testSplit)[0],:]
        w6_label = w6[1][:splitTrainTest(len(w6[1]), testSplit)[0],:]
        w9_label = w9[1][:splitTrainTest(len(w9[1]), testSplit)[0],:]
        
        h1_test = h1[0][-splitTrainTest(len(h1[0]), testSplit)[1]:,:,:]
        h2_test = h2[0][-splitTrainTest(len(h2[0]), testSplit)[1]:,:,:]
        h3_test = h3[0][-splitTrainTest(len(h3[0]), testSplit)[1]:,:,:]
        h4_test = h4[0][-splitTrainTest(len(h4[0]), testSplit)[1]:,:,:]
        h5_test = h5[0][-splitTrainTest(len(h5[0]), testSplit)[1]:,:,:]
        h6_test = h6[0][-splitTrainTest(len(h6[0]), testSplit)[1]:,:,:]
        h7_test = h7[0][-splitTrainTest(len(h7[0]), testSplit)[1]:,:,:]
        h8_test = h8[0][-splitTrainTest(len(h8[0]), testSplit)[1]:,:,:]
        h9_test = h9[0][-splitTrainTest(len(h9[0]), testSplit)[1]:,:,:]
        h10_test = h10[0][-splitTrainTest(len(h10[0]), testSplit)[1]:,:,:]
        w2_test = w2[0][-splitTrainTest(len(w2[0]), testSplit)[1]:,:,:]
        w3_test = w3[0][-splitTrainTest(len(w3[0]), testSplit)[1]:,:,:]
        w6_test = w6[0][-splitTrainTest(len(w6[0]), testSplit)[1]:,:,:]
        w9_test = w9[0][-splitTrainTest(len(w9[0]), testSplit)[1]:,:,:]
        
        h1_label2 = h1[1][-splitTrainTest(len(h1[0]), testSplit)[1]:,:]
        h2_label2 = h2[1][-splitTrainTest(len(h2[0]), testSplit)[1]:,:]
        h3_label2 = h3[1][-splitTrainTest(len(h3[0]), testSplit)[1]:,:]
        h4_label2 = h4[1][-splitTrainTest(len(h4[0]), testSplit)[1]:,:]
        h5_label2 = h5[1][-splitTrainTest(len(h5[0]), testSplit)[1]:,:]
        h6_label2 = h6[1][-splitTrainTest(len(h6[0]), testSplit)[1]:,:]
        h7_label2 = h7[1][-splitTrainTest(len(h7[0]), testSplit)[1]:,:]
        h8_label2 = h8[1][-splitTrainTest(len(h8[0]), testSplit)[1]:,:]
        h9_label2 = h9[1][-splitTrainTest(len(h9[0]), testSplit)[1]:,:]
        h10_label2 = h10[1][-splitTrainTest(len(h10[0]), testSplit)[1]:,:]
        w2_label2 = w2[1][-splitTrainTest(len(w2[0]), testSplit)[1]:,:]
        w3_label2 = w3[1][-splitTrainTest(len(w3[0]), testSplit)[1]:,:]
        w6_label2 = w6[1][-splitTrainTest(len(w6[0]), testSplit)[1]:,:]
        w9_label2 = w9[1][-splitTrainTest(len(w9[0]), testSplit)[1]:,:]
        
        

        trainingExamples.update({ (x,y) : np.concatenate((h1_train, h2_train, h3_train, h4_train, h5_train, h6_train, h7_train, h8_train, h9_train, h10_train, w2_train, w3_train, w6_train, w9_train)) })
        trainingLabels.update({ (x,y) : np.concatenate((h1_label, h2_label, h3_label, h4_label, h5_label, h6_label, h7_label, h8_label, h9_label, h10_label, w2_label, w3_label, w6_label, w9_label))})
        testExamples.update({ (x,y) : np.concatenate((h1_test, h2_test, h3_test, h4_test, h5_test, h6_test, h7_test, h8_test, h9_test, h10_test, w2_test, w3_test, w6_test, w9_test))})
        testLabels.update({ (x,y) : np.concatenate((h1_label2, h2_label2, h3_label2, h4_label2, h5_label2, h6_label2, h7_label2, h8_label2, h9_label2, h10_label2, w2_label2, w3_label2, w6_label2, w9_label2))})




scenarios=trainingExamples.keys()



#duplicate attempt
num_of_repts=10

#x_train=dict.fromkeys(scenarios)
for key, item in trainingExamples.items():
    trainingExamples[key] = np.tile(item, (num_of_repts,1,1))
    
#x_test=dict.fromkeys(scenarios)
#for key, item in testExamples.items():
#    testExamples[key] = np.tile(item, (num_of_repts,1,1))
    
#y_train=dict.fromkeys(scenarios)
for key, item in trainingLabels.items():
    trainingLabels[key] = np.tile(item, (num_of_repts,1))

#y_test=dict.fromkeys(scenarios)
#for key, item in testLabels.items():
#    testLabels[key] = np.tile(item, (num_of_repts,1))



#permutation
train_permutation=dict.fromkeys(scenarios)
test_permutation=dict.fromkeys(scenarios)

if scramble_data:
    print("Scramble data activated")
    for key, item in trainingExamples.items():
        train_permutation[key] = np.random.permutation(range(0, item.shape[0]))
        item = item[train_permutation[key][range(0, item.shape[0])], :, :]
    for key, item in trainingLabels.items():
        item = item[train_permutation[key][range(0, item.shape[0])], :]
        
#    for key, item in x_test.items():
#        test_permutation[key] = np.random.permutation(range(0, item.shape[0]))
#        item = item[test_permutation[key][range(0, item.shape[0])], 0:]
#    for key, item in y_test.items():
#        item = item[test_permutation[key]][range(0, item.shape[0])]
else:
    print("No scrambling, keep input data unchanged")




#training the RNN

numTrainingEpochs=10


for (k,v), (k2,v2) in zip(trainingExamples.items(), trainingLabels.items()):
    model_DNN = Sequential()

    
    model_DNN.add(LSTM(units=64, input_shape=(v.shape[1], v.shape[2])))
    model_DNN.add(Dense(10, activation='tanh'))
    model_DNN.add(Dense(nobs, activation='sigmoid'))
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model_DNN.compile(loss='mse', optimizer='Adam', metrics=["accuracy"])
    config = model_DNN.summary()
    history = model_DNN.fit(v, v2, validation_split=0.1, epochs=numTrainingEpochs)
    config = model_DNN.summary()
    model_DNN.save_weights("LSTM_weights/file_LSTM_weights_{}.h5".format(k))




