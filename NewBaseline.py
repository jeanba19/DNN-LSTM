#!/usr/bin/env python
# coding: utf-8



from InPT_trace_preparation import *
from NewSPtraffic import *
from random import *
from statsmodels.tsa.arima_model import ARIMA
import numpy

traces = [36,60,144]
percentiles = [70,80,90]
nobs=9



w1_InPT_trace=np.array(w1_InPT_trace)
SPT_w1 = np.array(SPT_w1)


import warnings
warnings.filterwarnings("ignore")


def generate_label(InPT_trace, SPT, trace_length=36, cap_value=80):

    cap=np.percentile(InPT_trace, cap_value)

    total = list(InPT_trace + SPT)

    opt_res=[]

    for i in range(n_time_slots):
        if total[i]<=cap:
            opt_res.append(0)
        else:
            opt_res.append(min(min(SPT[i] - (cap-InPT_trace[i]), SPT[i]), cap))

    opt_res_label = opt_res[trace_length:len(opt_res)]

    opt_res_label = np.array(opt_res_label).reshape((n_time_slots-trace_length, 1))

    return opt_res_label



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


def StartARIMAForecasting(Actual, P, D, Q, nobs):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)

    prediction = model_fit.forecast(nobs)[0]
    return prediction



def pb_identification(seq_length, inptORspt, nobs):
    pb = list()

    for t in range(n_time_slots-seq_length-nobs+1):
        TrainingData=inptORspt[t:t+seq_length]
        try:
            StartARIMAForecasting(TrainingData, 3,0,0, nobs)

        except:
            pb.append(t)
    return pb






def generate_prediction(seq_length,inptOrspt, nobs):
    
    Predictions=list()
    
    compteur = 0

    for t in range(n_time_slots-seq_length-nobs+1):

        TrainingData=inptOrspt[t:t+seq_length]
        TestData=inptOrspt[t+seq_length:t+seq_length+nobs]
        
        #forecast part
        try:
            Prediction = StartARIMAForecasting(TrainingData, 3,0,0, nobs)
        except:
            Prediction = "problem"
            compteur+=1

        Predictions.append(Prediction)
        
        
    return np.array(Predictions)



def problem_length(pbSPT, pbInPT):
    
    pb=pbSPT+pbInPT
    pb=list(set(pb))
    return len(pb)




PbInpt={}
for x in traces:
    pb = pb_identification(x, w1_InPT_trace, nobs)
    PbInpt.update( { x : pb } )

PbSpt={}
for x in traces:
    pb = pb_identification(x, SPT_w1, nobs)
    PbSpt.update( { x : pb } )


def reservation_pred(seq_length, sptPred, inptPred, cap, pbSPT, pbInPT):
     

    res = numpy.zeros(shape=(n_time_slots-seq_length-nobs+1-problem_length(pbSPT, pbInPT),nobs))
    
    i=0

    while i < (n_time_slots - seq_length - nobs + 1):
        if i in pbSPT or i in pbInPT:
            i+=1
        else:
            
            for t in range(nobs):
                if sptPred[i][t] + inptPred[i][t] <= cap:
                    res[i][t] = 0
                else:
                    res[i][t] = min(min(sptPred[i][t] - (cap - inptPred[i][t]), sptPred[i][t]), cap)
            
            i+=1

    return res
        
    




def error(pred, label):
    return pred - label





def Overres(array):
    return np.sum(array[array>0]) / len(array)




def Underres(array):
    return np.sum(array[array<0]) / len(array)




def MSE(array):
    return np.dot(array, array) / len(array)




def AvgAmountOverRes(array):
    compteur=0
    somme=0
    for j in array:
        if j>0:
            somme = somme + j
            compteur += 1
        else:
            somme = somme
            compteur = compteur

    return somme/compteur




def AvgAmountUnderRes(array):
    compteur=0
    somme=0
    for j in array:
        if j<0:
            somme = somme + j
            compteur += 1
        else:
            somme = somme
            compteur = compteur

    return somme/compteur




Metrics={}
for x in traces:
    for y in percentiles:
        metrics = ({'mean_over_reservation':[Overres(error(reservation_pred(x,generate_prediction(x, SPT_w1, nobs),generate_prediction(x, w1_InPT_trace, nobs), np.percentile(w1_InPT_trace,y),PbSpt[x],PbInpt[x])), labelForMultipleStepsPrediction(generate_label(w1_InPT_trace,SPT_w1,x,y)))],'mean_under_reservation':[Underres(error(reservation_pred(x,generate_prediction(x, SPT_w1, nobs),generate_prediction(x, w1_InPT_trace, nobs), np.percentile(w1_InPT_trace,y),PbSpt[x],PbInpt[x])), labelForMultipleStepsPrediction(generate_label(w1_InPT_trace,SPT_w1,x,y)))],'amountOver':[AvgAmountOverRes(error(reservation_pred(x,generate_prediction(x, SPT_w1, nobs),generate_prediction(x, w1_InPT_trace, nobs), np.percentile(w1_InPT_trace,y),PbSpt[x],PbInpt[x])), labelForMultipleStepsPrediction(generate_label(w1_InPT_trace,SPT_w1,x,y)))],'amountUnder':[AvgAmountUnderRes(error(reservation_pred(x,generate_prediction(x, SPT_w1, nobs),generate_prediction(x, w1_InPT_trace, nobs), np.percentile(w1_InPT_trace,y),PbSpt[x],PbInpt[x])), labelForMultipleStepsPrediction(generate_label(w1_InPT_trace,SPT_w1,x,y)))],'MSE':[MSE(error(reservation_pred(x,generate_prediction(x, SPT_w1, nobs),generate_prediction(x, w1_InPT_trace, nobs), np.percentile(w1_InPT_trace,y),PbSpt[x],PbInpt[x])), labelForMultipleStepsPrediction(generate_label(w1_InPT_trace,SPT_w1,x,y)))]})
        Metrics.update( {str((x,y)) : metrics } )
import json
with open('results_baseline_3_0_0.json') as f:
    json.dump(Metrics, f)

