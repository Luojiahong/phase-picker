#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINALISED SCRIPT TO LOAD AN AUTOPICKER, WINDOW [USING CENTRE METHOD] AND THEN 
MAKE PREDICTIONS ON A CONTINUOUS STREAM USING THE TRAINED CNN

@author: J.Woollam        June 2018
"""

""" Lib dependancies """
#==============================================================================
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import obspy as obs
from obspy.signal.filter import bandpass
from scipy.signal import butter, lfilter
import time
import pathlib


"""load autopicker"""
#==============================================================================
# load model
model_name='CNN_2800_contstd_xtrapicks_gaussnorm_30_sigmatest_0p25.h5'
#model_name = 'CNN_2400_5050_500winpbatch3comp_meanstd.h5'
loaded_model = load_model('./models/'+model_name)
print("Loaded model from disk")

# Need to define size of window used to make predictions
window_size = 2800

"""Define functions to read in and manipulate .mseed files"""
#==============================================================================

'Functions to Bandpass filter'
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

'Functions to perform certain types of normalisation'
def norm_meanstd(window):
    #Function to normalise by taking the mean of the std of all components
    std1 = np.std(window[:,0],axis=0)
    std2 = np.std(window[:,1],axis=0)
    std3 = np.std(window[:,2],axis=0)
    
    window_norm = window/np.mean([std1,std2,std3])
    return window_norm

def norm_individualstd(window):
    #Function to normalise individual components by the std
    window[:,0] = window[:,0]/np.std(window[:,0],axis=0)
    window[:,1] = window[:,1]/np.std(window[:,1],axis=0)
    window[:,2] = window[:,2]/np.std(window[:,2],axis=0)
    
    window_norm = window
    return window_norm

def test_window_wstep(timeseries, timestep, window_size):
        
    for i in range(0,
                   (int(len(timeseries)/timestep))*timestep,
                   timestep):
        #print(i)
        window = timeseries[i:i+window_size]
        
        #if len(window) == window_size:            
        yield window


"""Read in and process continuous data"""
#==============================================================================
start_time = time.time()
days2pred = [str(i) for i in range(120,129)]
#IN_stats = ['0'+str(i) for i in range(1,6)]
IN_stats = ['01','03']
days2pred = ['120','121']

for stat in IN_stats:
    for day in days2pred:
        print('\nPREDICTING ON DAY --- {} ---'.format(day))
        # Load in continuous str
        st = obs.read('../Data/cont_strs/IN'+stat+'/raw_data/HHE.D/DG.IN'+stat+'..HHE.D.2014.'+day)
        trE=st[0]
        st = obs.read('../Data/cont_strs/IN'+stat+'/raw_data/HHN.D/DG.IN'+stat+'..HHN.D.2014.'+day)
        trN=st[0]
        st = obs.read('../Data/cont_strs/IN'+stat+'/raw_data/HHZ.D/DG.IN'+stat+'..HHZ.D.2014.'+day)
        trZ=st[0]
    
        #merge components into single list
        cont_tr = [trE.data, trN.data, trZ.data]
        cont_tr = np.array(cont_tr,dtype=float)
        cont_tr = np.transpose(cont_tr)
        
        #detrend
        cont_tr[:,0] = cont_tr[:,0] - np.mean(cont_tr[:,0])
        cont_tr[:,1] = cont_tr[:,1] - np.mean(cont_tr[:,1])
        cont_tr[:,2] = cont_tr[:,2] - np.mean(cont_tr[:,2])
        
        #Bp fiter
        cont_tr[:,0] = butter_bandpass_filter(cont_tr[:,0],2,25,100)       
        cont_tr[:,1] = butter_bandpass_filter(cont_tr[:,1],2,25,100)    
        cont_tr[:,2] = butter_bandpass_filter(cont_tr[:,2],2,25,100)                
        
        tr_segments=[]
        pr_segments=[]
             
        
        
        # Get window generator for entire trace
        window_gen = test_window_wstep(cont_tr, int(window_size/2),window_size)

        windowed_data = []
        endwindow=[]
        # Window entire trace
        for window in window_gen:
            #window = window/np.amax(window)
            window = norm_meanstd(window[:])
            windowed_data.append(window)
            
        windowed_data = windowed_data[:-1]
        windowed_data = np.array(windowed_data)
        
        # Make predictions on windows
        pred_windows = loaded_model.predict(windowed_data)
        
        tr_centres=[]
        pr_centres=[]
        
        # Get centre of windowed data
        for i in range(len(pred_windows)):
            centre_start = int((window_size/4))
            centre_end = int((window_size/4)*3)
            tr_centres.append(windowed_data[i,centre_start:centre_end,:])
            pr_centres.append(pred_windows[i,centre_start:centre_end,:])
                
        tr_centres = np.array(tr_centres)
        pr_centres = np.array(pr_centres)        
        
        #Reshape the centre portions of the list to match input data
        tr_centres = np.reshape(tr_centres,(np.shape(tr_centres)[0]*np.shape(tr_centres)[1],3))
        pr_centres = np.reshape(pr_centres,(np.shape(pr_centres)[0]*np.shape(pr_centres)[1],3))    
        
        # Pad with zeros to remove windowing effects at start and end of 
        # continuous trace        
        padstart = np.zeros((int(window_size/4),3),dtype=float)
        padend = np.zeros((len(cont_tr)-(len(tr_centres)+int(window_size/4)),3),dtype=float)
        tr_centres = np.concatenate((padstart,tr_centres,padend),axis=0)    
        pr_centres = np.concatenate((padstart,pr_centres,padend),axis=0)
        
        # Save arrays of predictions
        output_dir = './predictions/predictions_as_arrays/CNN2800/'
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        print('Saving arrays of predictions --- STAT: IN{} --- JDAY: {}'.format(stat, day))
        np.save(output_dir + 'DG.IN'+stat+'..HHE.D.2014.'+day+'_preds',pr_centres)

        #print('Saving filtered stream       --- STAT: IN{} --- JDAY: {}'.format(stat, day))
        #np.save(output_dir + 'DG.IN'+stat+'..HHE.D.2014.'+day+'_stream',tr_centres)        
        
        
        
        
        