import numpy as np
import math
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, CuDNNLSTM
from sklearn import preprocessing
import random
import h5py
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from InPT_trace_preparation import *
from NewSPtraffic import *
from BS_choice import *



spt_vector_outofrun = np.array(sim_input["w1"][1])

inpt_vector_outofrun = np.array(sim_input["w1"][0])

cap_outofrun = sim_input["w1"][2]

cap_outofrun_2 = np.percentile(inpt_vector_outofrun, 80)

traces=[6,18,36,60]
percentiles=[70,80,90]
ANN_sample_dim=slot_length_per_hour*sample_length_in_hours
nobs=9





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





def previous_SPT_func():
    res=np.roll(spt_vector_outofrun,1)
    res[0]=max(spt_vector_outofrun)
    return res





def generate_label(cap_value=80):
    opt_res_vector_t0=np.zeros((n_time_slots,1))
    for t in range(n_time_slots):
        spt = spt_vector_outofrun[t]
        inpt = inpt_vector_outofrun[t]
        if cap_value==80:
            cap = cap_outofrun   #by default, we choose the capacity as the 80th percentile of eNb traffic, to have normal traffic conditions
        else:
            cap = np.percentile(inpt_vector_outofrun, cap_value)
        if inpt + spt <= cap:
            opt_res = 0
        else:
            opt_res = min (min( spt - (cap - inpt), spt), cap)

        opt_res_vector_t0[t]=opt_res

    return labelForMultipleStepsPrediction(opt_res_vector_t0, nobs=9)



def splitTrainTest(datalength, testSplit):
    num_train_samples = int(datalength*(1-testSplit))
    num_test_samples = datalength - num_train_samples
    return num_train_samples, num_test_samples

def AvgAmountOverRes(array):
    compteur=0
    somme=0
    for j in array:
        if j>0:
            somme = somme + j
            compteur+=1
        else:
            somme = somme
            compteur = compteur
    if compteur==0:
        return 0
    else:
        return somme/compteur

def AvgAmountUnderRes(array):
    compteur=0
    somme=0
    for j in array:
        if j<0:
            somme = somme + j
            compteur+=1
        else:
            somme = somme
            compteur = compteur
    if compteur==0:
        return 0
    else:
        return somme/compteur

def run_simulation(res_method, eNb_choice, cap_value=80, seq_length=36):

    #input dimensions
    net_type=['DNN','RNN']
    numFeatures=4

    num_of_repts=10
    
    dim_in=seq_length*4


    transient_length=50
    day_max_length=144   

    #initialize vectors
    spt_vector = np.zeros(n_time_slots-nobs+1)
    inp_vector = np.zeros(n_time_slots-nobs+1)
    ser_vector = np.zeros(n_time_slots-nobs+1)
    uns_vector = np.zeros(n_time_slots-nobs+1)
    bser_vector = np.zeros(n_time_slots-nobs+1)
    res_vector = np.zeros((n_time_slots-nobs+1,nobs))

    x_in = np.zeros((dim_in,n_time_slots))
    opt_res_vector_t0 = np.zeros((n_time_slots-nobs+1,nobs))
    res_error_vector = np.zeros((n_time_slots-nobs+1,nobs))
    x_t_in_vect = np.zeros((dim_in,1))

    print("The reservation method used is: {}".format(res_method))
    print(" ")

    if res_method == 0:
        # Generate DNN
        model_DNN = Sequential()
        
        model_DNN.add(Dense(50, activation = 'tanh', input_dim = seq_length*numFeatures))

        model_DNN.add(Dense(10, activation = 'tanh'))
        model_DNN.add(Dense(nobs, activation = 'sigmoid'))
        model_DNN.compile(loss = 'mse', optimizer = 'Adam', metrics = ["accuracy"])
        model_DNN.load_weights("DNN_weights/file_DNN_weights_({}, {}).h5".format(seq_length,cap_value), by_name=False)

        transient_length = max(transient_length,day_max_length)
    elif res_method == 1:
        # Generate LSTM
        model_DNN = Sequential()
        model_DNN.add(LSTM(units=64, input_shape=(seq_length, numFeatures)))
        model_DNN.add(Dense(10, activation='tanh'))
        model_DNN.add(Dense(nobs, activation='sigmoid'))
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        model_DNN.compile(loss = 'mse', optimizer = 'Adam', metrics = ["accuracy"])
        model_DNN.load_weights("LSTM_weights/file_LSTM_weights_({}, {}).h5".format(seq_length, cap_value), by_name=False)

        transient_length = max(transient_length,day_max_length)
    # end IF   



    max_expected_spt_traffic=max(sim_input[eNb_choice][1])
    resVal_zero=0
    resVal_max=max_expected_spt_traffic

    for t in range(0,n_time_slots-nobs+1):

        ##################################
        # CHOOSE ACTION

        if res_method == 0:   # DNN
            if t < seq_length:
                res = np.full((1,nobs),max_expected_spt_traffic)
            else:
                
                res = model_DNN.predict(x_t_in_vect)
                
        elif res_method == 1: # LSTM
            if t < seq_length:
                res = np.full((1,nobs),max_expected_spt_traffic)
            else:
                
                x_t_in_vect = x_t_in_vect.reshape(1,numFeatures, seq_length)
                x_t_in_vect = x_t_in_vect.transpose(0,2,1)
                res = model_DNN.predict(x_t_in_vect)
                
        #elif res_method == 2 | res_method == 3:    # constant zero reservation, constant max reservation
        elif res_method == 2:
            res = np.full((1,nobs),resVal_zero)
        elif res_method == 3:  # constant zero reservation, constant max reservation
            res = np.full((1,nobs),resVal_max)
        elif res_method == 4:   #previous SPT
            res = np.full((1,nobs),previous_SPT_func()[t])

  

        else:
            res = np.full((1,nobs),0)
        # end IF

        ##################################
        # GENERATE TRAFFIC
        # generate SP traffic
        spt = spt_vector_outofrun[t]
        # generate INPT traffic
        inpt = inpt_vector_outofrun[t]

        if cap_value==80:
            cap = cap_outofrun   #by default, we choose the capacity as the 80th percentile of eNb traffic, to have normal traffic conditions
        else:
            cap = np.percentile(inpt_vector_outofrun, cap_value) #we give ourselves the right to choose cap_value lower or higher to test the solution over different traffic conditions



        ##################################
        # ASSESS DELIVERED TRAFFIC

        if  inpt + spt <= cap:
            ser = spt
            uns = 0
            lo = inpt + spt
            bser = max(spt-res[0][0],0)

        else:
            ser = max( min(spt,res[0][0] ),cap-inpt )
            uns = max( min(spt-res[0][0],spt-cap+inpt ),0)
            lo = cap
            bser = max(cap - inpt - res[0][0],0)


       

        res_error_vector[t] = res - label[t]

        ##################################
        # SAVE DATA
        spt_vector[t] = spt
        inp_vector[t] = inpt
        ser_vector[t] = ser
        uns_vector[t] = uns
        bser_vector[t] = bser
        
        res_vector[t] = res
        opt_res_vector_t0[t] = label[t]


        if t >= (seq_length - 1):
            tmp = np.array( np.concatenate( (spt_vector[ (t-seq_length+1):(t+1) ], uns_vector[ (t-seq_length+1):(t+1) ], bser_vector[ (t-seq_length+1):(t+1) ], res_vector[ (t-seq_length+1):(t+1) ][:,0] ) ))
            x_t_in_vect = np.reshape(tmp,(1,seq_length*numFeatures))

   


    print("Loop Completed")

    idx_all = np.array(range(0,spt_vector.size))



    #print("Mean percentage of unserved traffic")
    mean_perc_uns_traffic = np.mean(  uns_vector[ np.logical_and( spt_vector>0,  idx_all > transient_length ) ] / spt_vector[ np.logical_and(spt_vector>0,  (idx_all > transient_length) ) ] *100)
    #print(mean_perc_uns_traffic)

    #print("Percentage of denial of service")
    perc_denial_of_traffic = np.sum(  ser_vector[ np.logical_and( spt_vector>0,  idx_all > transient_length ) ] <= 0) / np.sum( np.logical_and( spt_vector>0,  idx_all > transient_length ) )*100
    #perc_denial_of_traffic = np.sum(  uns_vector[ np.logical_and( spt_vector>0,  idx_all > transient_length ) ] >= 0) / np.sum( np.logical_and( spt_vector>0,  idx_all > transient_length ) )*100
    #print(perc_denial_of_traffic)

    tmp = res_error_vector[ idx_all > transient_length ]
    mean_over_reservation = [np.sum( tmp[:,i][ tmp[:,i] >0 ]  ) / np.sum(idx_all > transient_length) for i in range(nobs)]

    tmp = res_error_vector[ idx_all > transient_length ]
    mean_under_reservation = [np.sum( tmp[:,i][ tmp[:,i] <0 ]  ) / np.sum(idx_all > transient_length) for i in range(nobs)]

    #print("Mean over-reservation | reservation is made")
    tmp = res_error_vector[ idx_all > transient_length ]
    mean_over_reservation_cond = [np.sum( tmp[:,i][ tmp[:,i] >0 ]  ) / np.sum(opt_res_vector_t0[:,i][ idx_all > transient_length]>0 ) for i in range(nobs)]

    #print("Mean under-reservation | reservation is made")
    tmp = res_error_vector[ idx_all > transient_length ]
    mean_under_reservation_cond = [np.sum( tmp[:,i][ tmp[:,i] <0 ]  ) / np.sum(opt_res_vector_t0[:,i][ idx_all > transient_length]>0 ) for i in range(nobs) ]

    tmp = res_error_vector[ idx_all > transient_length ]
    amountOver = [ AvgAmountOverRes(tmp[:,i]) for i in range(nobs) ]

    tmp = res_error_vector[ idx_all > transient_length ]
    amountUnder = [ AvgAmountUnderRes(tmp[:,i]) for i in range(nobs) ]

    tmp = res_error_vector[ idx_all > transient_length ]
    reservation_MSE = [ np.dot(tmp[:,i],tmp[:,i]) / tmp.shape[0] for i in range(nobs) ]



    #ser_vector_norm = ser_vector/max_expected_spt_traffic
    #mean_ser_traffic = np.mean(  ser_vector_norm[ idx_all > transient_length ])
    #mean_ser_traffic = np.mean(  ser_vector[ idx_all > transient_length ])

    #res_vector_norm = res_vector/max_expected_spt_traffic
    #mean_reservation = np.mean(  res_vector_norm[ idx_all > transient_length ])
    #mean_reservation = np.mean(  res_vector[ idx_all > transient_length ])

    #uns_vector_norm = uns_vector/max_expected_spt_traffic
    #mean_uns_traffic = np.mean(  uns_vector_norm[ idx_all > transient_length ])
    #mean_uns_traffic = np.mean(  uns_vector[ idx_all > transient_length ])

    #bser_vector_norm = bser_vector/max_expected_spt_traffic
    #mean_bser_traffic = np.mean(  uns_vector_norm[ idx_all > transient_length ])
    #mean_bser_traffic = np.mean(  bser_vector[ idx_all > transient_length ])

    #results = [mean_perc_uns_traffic, perc_denial_of_traffic, mean_over_reservation, mean_under_reservation, mean_over_reservation_cond, mean_under_reservation_cond, reservation_MSE, mean_ser_traffic, mean_reservation, mean_uns_traffic, mean_bser_traffic]

    metrics = ({'amountOver':amountOver, 'amountUnder':amountUnder, 'MSE':reservation_MSE, 'mean_over_reservation':mean_over_reservation, 'mean_under_reservation':mean_under_reservation, 'mean_over_reservation_cond':mean_over_reservation_cond, 'mean_under_reservation_cond':mean_under_reservation_cond})
    #metrics = [mean_over_reservation_1step, mean_over_reservation_2steps, mean_over_reservation_3steps, mean_over_reservation_4steps, mean_under_reservation_1step, mean_under_reservation_2steps, mean_under_reservation_3steps, mean_under_reservation_4steps, reservation_MSE, reservation_MSE_2steps, reservation_MSE_3steps, reservation_MSE_4steps]
    return metrics

Metrics={}
for a in [0,1,2,3,4]:
    for x in traces:
        for y in percentiles:
            cap_outofrun_2 = np.percentile(inpt_vector_outofrun, y)
            label = generate_label(y)
            Metrics.update({ str((a,x,y)) : run_simulation(res_method=a, eNb_choice="w1", cap_value=y, seq_length=x)})



import json
with open('results_ML.json', 'w') as f:
    json.dump(Metrics, f)



#import json
#with open('results.json') as f:
#    a = json.load(f)
