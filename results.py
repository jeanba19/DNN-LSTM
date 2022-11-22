#!/usr/bin/env python
# coding: utf-8




import json
def import_file(file_name):
    with open(file_name) as f:
        getResults = json.load(f)
    return getResults    





from collections import MutableMapping 
def convert_flatten(nestedDictionary, parent_key ='', sep =','): 
    items = [] 
    for k, v in nestedDictionary.items(): 
        new_key = parent_key + sep + k if parent_key else k 
  
        if isinstance(v, MutableMapping): 
            items.extend(convert_flatten(v, new_key, sep = sep).items()) 
        else: 
            items.append((new_key, v)) 
    return dict(items)




import pandas as pd
def convert_to_dataframe_v2(flattenDictionary):
    tmp=pd.DataFrame((list(flattenDictionary.items())))
    L=[]
    for row in range(len(tmp[0])):
        L.append(tmp[0][row].split(','))
        Keys=pd.DataFrame(L)
        temp = pd.concat([Keys, tmp], axis=1)
        temp.columns = ['method', 'seq', 'cap', 'step', 'cc', 'value']
        temp=temp.drop(['cc'], axis=1)
        temp=temp.replace(' 70)', 70)
        temp=temp.replace(' 80)', 80)
        temp=temp.replace(' 90)', 90)
        temp=temp.replace(' 18', 18)
        temp=temp.replace(' 36', 36)
        temp=temp.replace(' 6', 6)
        temp=temp.replace(' 60', 60)
    return temp 




def finalize_dataframe(dataframe):
    
    _=dataframe.loc[dataframe["method"]=='(0']
    del _['method']
    _.columns = ['seq', 'cap', 'step', 'DNN']


    __=dataframe.loc[dataframe["method"]=='(1']
    __=__.reset_index()
    del __['method']
    del __['index']
    __.columns = ['seq', 'cap', 'step', 'LSTM']
    

    ___=dataframe.loc[dataframe["method"]=='(2']
    ___=___.reset_index()
    del ___['method']
    del ___['index']
    ___.columns = ['seq', 'cap', 'step', 'constzero']


    ____=dataframe.loc[dataframe["method"]=='(3']
    ____=____.reset_index()
    del ____['method']
    del ____['index']
    ____.columns = ['seq', 'cap', 'step', 'constmax']


    _____=dataframe.loc[dataframe["method"]=='(4']
    _____=_____.reset_index()
    del _____['method']
    del _____['index']
    _____.columns = ['seq', 'cap', 'step', 'previousSPT']


    return pd.concat([_,__['LSTM'],___['constzero'],____['constmax'],_____['previousSPT']], axis=1)



method_comp['value0']=method_comp['value0'].abs()




temp=finalize_dataframe(convert_to_dataframe_v2(convert_flatten(import_file('results_ML.json'))))
dataframeDict.update( { 'ML' : temp } )




import json
with open('results_baseline.json') as f:
    bsl = json.load(f)




from collections import MutableMapping 
def convert_flatten(d, parent_key ='', sep =','): 
    items = [] 
    for k, v in d.items(): 
        new_key = parent_key + sep + k if parent_key else k 
  
        if isinstance(v, MutableMapping): 
            items.extend(convert_flatten(v, new_key, sep = sep).items()) 
        else: 
            items.append((new_key, v)) 
    return dict(items)




bslb=convert_flatten(bsl)



import pandas as pd
bslBD=pd.DataFrame((list(bslb.items())))



def convert_to_dataframe(frame):
    L=[]
    for row in range(len(frame[0])):
        L.append(frame[0][row].split(','))
        Keys=pd.DataFrame(L)
        tmp = pd.concat([Keys, frame], axis=1)
        tmp.columns = ['seq', 'cap', 'step', 'cc', 'value']
        tmp=tmp.drop(['cc'], axis=1)
        tmp=tmp.replace(' 70)', 70)
        tmp=tmp.replace(' 80)', 80)
        tmp=tmp.replace(' 90)', 90)
        #tmp=tmp.replace('(18', 18)
        tmp=tmp.replace('(36', 36)
        #tmp=tmp.replace('(6', 6)
        tmp=tmp.replace('(60', 60)
        tmp=tmp.replace('reservation_MSE', 'MSE')
    return tmp   




final_bsl = convert_to_dataframe(bslBD)




def reformat_for_csv(series, method):
    L=[]
    for x in series[method]:
        L.append(x)
    import numpy as np
    M=np.array(L)
    import pandas as pd
    Vectors=pd.DataFrame(M)
    result = pd.concat([series, Vectors], axis=1)
    result = result.drop([method], axis=1)
    result = result.drop('index', axis=1)
    result['method']=method
    return result




def reformat_to_csv(dataframe, method):
    L = dataframe[method][0] 
    df = pd.DataFrame({method:L})
    return df




import numpy as np
def get_sequence_perf(dataframe, method, metric, capacity, seq):
    data = dataframe[dataframe['cap']==capacity]
    data = data[data['step'] == metric]
    data = data[data['seq'] == seq]
    series = data[method]
    seq_perf=series.tolist()
    
    
    return np.mean(seq_perf[0])
#for the baseline the best sequence is 60



traces=[6,18,36,60]
def best_sequence(dataframe, method, metric, capacity):
    best_perf=float("inf")
    for x in traces:
        seq_perf = get_sequence_perf(dataframe, method, metric, capacity, x)
        if seq_perf < best_perf:
            best_perf=seq_perf
            best_seq=x
    return best_perf, best_seq 




def overall_best_sequence(dataframes, method, metric, capacity):
    overall_best_perf=float("inf")
    for k,v in dataframes.items():
        overall_seq_perf = best_sequence(v, method, metric, capacity)[0]
        if overall_seq_perf < overall_best_perf:
            overall_best_perf=overall_seq_perf
            overall_best_seq = best_sequence(v, method, metric, capacity)[1]
            overall_best_hyper = k
    return overall_best_perf, overall_best_seq, overall_best_hyper




traces=[6,18,36,60]
percentiles=[70,80,90]
def hyper_perf(dataframe, method, metric):
    perfs=[]
    for w in percentiles:
            seq_perf = best_sequence(dataframe, method, metric, w)[0]
            perfs.append(seq_perf)
    return np.mean(perfs)



def choose_hyper(dataframes, method, metric):
    best_hyper_perf = float("inf")
    for k,v in dataframes.items():
        hyper_performance = hyper_perf(v, method, metric)
        if hyper_performance < best_hyper_perf:
            best_hyper_perf = hyper_performance
            best_hyper = k
    return best_hyper_perf, best_hyper




choose_hyper(dataframeDict, 'LSTM', 'MSE')



overall_best_sequence(dataframeDict, 'DNN', 'MSE', 80)




def overall_best_sequence_v2(dataframes, method, metric, capacity):
    overall_best_perf=float("inf")
    for x in dataframes:
        overall_seq_perf = best_sequence(x, method, metric, capacity)[0]
        if overall_seq_perf < overall_best_perf:
            overall_best_perf=overall_seq_perf
            overall_best_seq = best_sequence(x, method, metric, capacity)[1]
    return overall_best_perf, overall_best_seq




overall_best_sequence_v2(dataframes, 'DNN', 'MSE', 70)




best_sequence(dataframes[0], 'DNN', 'MSE', 70)




best_sequence(dataframeDict["('tts', '01')"], 'DNN', 'MSE', 70)




def createDataframe_forCsv(dataframe, metric, capacity):
    

        
    solDNN=dataframe[dataframe['cap']==capacity]
    solDNN=solDNN[solDNN['step']==metric]
    solDNN=solDNN[solDNN['seq']==best_sequence(methods[0],capacity)]
    solDNN = solDNN[['seq','cap', 'step', methods[0]]]
    solDNN=solDNN.reset_index()
    #solDNN=reformat_for_csv(solDNN, methods[0])
    solDNN = reformat_to_csv(solDNN, methods[0])
    
    solLSTM=dataframe[dataframe['cap']==capacity]
    solLSTM=solLSTM[solLSTM['step']==metric]
    solLSTM=solLSTM[solLSTM['seq']==best_sequence(methods[1],capacity)]
    solLSTM = solLSTM[['seq','cap', 'step', methods[1]]]
    solLSTM=solLSTM.reset_index()
    #solLSTM=reformat_for_csv(solLSTM, methods[1])
    solLSTM = reformat_to_csv(solLSTM, methods[1])

    bsl=final_bsl[final_bsl['cap']==capacity]
    bsl=bsl[bsl['step']==metric]
    bsl=bsl[bsl['seq']==60]
    bsl=bsl.reset_index()
    bsl = reformat_to_csv(bsl, 'value')
    
    #bsl=reformat_for_csv(bsl, 'value')
    #return sol.to_csv(method+metric+str(capacity)+'.csv')
    #return pd.concat([sol,bsl])
    return pd.concat([solDNN,solLSTM,bsl],axis=1)




def createCsv(metric, capacity):
    return createDataframe_forCsv(metric, capacity).to_csv(metric+str(capacity)+'.csv')




metrics=['MSE','amountOver','amountUnder']
capacities=[70,80,90]
methods=['DNN', 'LSTM', 'constzero', 'constmax', 'previousSPT', 'value']




def graph_ylabel(metric):
    if metric=='MSE':
        title = r'\textit{mean square error}'
    elif metric=='amountOver':
        title = r'\textit{average over reservation}'
    else:
        title = r'\textit{average under reservation}'
    return title




def graph_title(capacity):
    if capacity==70:
        title = r'\textit{high traffic}'
    elif capacity==80:
        title = r'\textit{regular traffic}'
    else:
        title = r'\textit{low traffic}'
    return title




import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def method_compare(dataframe_DNN, dataframe_LSTM, metric, capacity):
    
    
    
    #dataframe_DNN['DNN']=dataframe_DNN['DNN'].abs()
    
    #dataframe_LSTM['LSTM']=dataframe_LSTM['LSTM'].abs()
    
    bsltrend=final_bsl[final_bsl['step'] == metric]
    
    bsltrend=bsltrend[bsltrend['cap']==capacity]
    
    DNNtrend=dataframe_DNN[dataframe_DNN['step'] == metric]
    
    DNNtrend=DNNtrend[DNNtrend['cap']==capacity]
    
    LSTMtrend=dataframe_LSTM[dataframe_LSTM['step'] == metric]
    
    LSTMtrend=LSTMtrend[LSTMtrend['cap']==capacity]

    nobs=9

    x=range(1,nobs+1)
    
    from matplotlib import rc
    rc('text', usetex=True)
    import numpy
    import pylab
    
    plt.tight_layout()
    
    fig=pylab.plot(x, list(bsltrend.loc[bsltrend['seq']==60 ]['value'])[0], '.', label=r'\textit{baseline}', color='green')
    
    fig=pylab.plot(x, list(DNNtrend.loc[DNNtrend['seq']==best_sequence(dataframe_DNN,'DNN', metric, capacity)[1] ]['DNN'])[0], '.', label=r'\textit{DNN}', color='red')
    fig=pylab.plot(x, list(LSTMtrend.loc[LSTMtrend['seq']==best_sequence(dataframe_LSTM,'LSTM', metric, capacity)[1] ]['LSTM'])[0], '.', label=r'\textit{LSTM}', color='blue')

    fig=pylab.legend(loc='best', prop={'size': 15}, frameon=True)
    fig=pylab.xlabel(r'\textit{number of steps ahead}', fontsize=20)
    fig=pylab.ylabel(graph_ylabel(metric), fontsize=20)
    fig=pylab.title(graph_title(capacity), fontsize=22)
    fig=pylab.xticks(numpy.arange(1, nobs+1, step=1), fontsize=16)
    fig=pylab.yticks(fontsize=16)
    #fig.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    # calc the trendline

    z2 = numpy.polyfit(x, list(bsltrend.loc[bsltrend['seq']==60 ]['value'])[0], 4)
    p2 = numpy.poly1d(z2)

    z3 = numpy.polyfit(x, list(DNNtrend.loc[DNNtrend['seq']==best_sequence(dataframe_DNN,'DNN', metric, capacity)[1] ]['DNN'])[0], 4)
    p3 = numpy.poly1d(z3)

    z4 = numpy.polyfit(x, list(LSTMtrend.loc[LSTMtrend['seq']==best_sequence(dataframe_LSTM,'LSTM', metric, capacity)[1] ]['LSTM'])[0], 4)
    p4 = numpy.poly1d(z4)

    fig=pylab.plot(x,p2(x),"g-")
    fig=pylab.plot(x,p3(x),"r-")
    fig=pylab.plot(x,p4(x),"b-")

    return fig



method_compare(dataframeDict["('tts', '001')"],dataframeDict["('tts', '0001')"], 'amountUnder', 90)
plt.savefig('test.pdf', bbox_inches='tight')




#fig = plt.figure(figsize=(8, 5), tight_layout=False)

for x in capacities:
    plt.gcf().subplots_adjust(bottom=0.15)
    method_compare(dataframeDict["('tts', '001')"],dataframeDict["('tts', '0001')"], 'MSE', x)
    plt.savefig('MSE'+str(x)+'.pdf', bbox_inches='tight')
    plt.clf()




for x in capacities:
    plt.gcf().subplots_adjust(bottom=0.15)
    method_compare(dataframeDict["('tts', '001')"],dataframeDict["('tts', '0001')"], 'amountUnder', x)
    plt.savefig('Under'+str(x)+'.pdf', bbox_inches='tight')
    plt.clf()




for x in capacities:
    plt.gcf().subplots_adjust(bottom=0.15)
    method_compare(dataframeDict["('tts', '001')"],dataframeDict["('tts', '0001')"], 'amountOver', x)
    plt.savefig('Over'+str(x)+'.pdf', bbox_inches='tight')
    plt.clf()




