# DNN-LSTM


Here is the solution presented in Chapter 3 of the thesis.
We have 2 py files to train the DNN and LSTM models for different scenarios (the length of the sequence to predict next $h$ reservations, the traffic conditions) => (traces, percentiles)

We prepare the data with InPT_trace_preparation.py and New_SPTraffic.py files

We simulate the online environment with the py file simulation, we store the results in json files.

We print out the graphs based on the results in results.py
