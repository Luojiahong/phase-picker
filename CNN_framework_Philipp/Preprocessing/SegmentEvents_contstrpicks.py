#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to segment events which have been picked in the continous streams

@author: jack
"""

""" Lib dependancies """
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass
import obspy as obs
import pathlib


""" Segment events [picked on continuous str]"""
#==============================================================================
# Get IDs for continuous str... (1 contnuous str contains 1 day of data)
days2segment = [i for i in range(120,130)]
days2segment.remove(126)

output_dir = '../Data/prep_data2train/temp/segmented_events/'

# Loop trhough streams and segment picked events
for day in days2segment:
    
    # Load labels for each day
    a = np.load('../Data/prep_data2train/temp/labels/day'+str(day)+'.npy')
    
    # Load in raw data for each day
    b = np.load('../Data/prep_data2train/temp/data/day'+str(day)+'.npy')
    b = np.transpose(b)
    
    # Load in indexes of events
    newindexes=np.load('../Data/prep_data2train/temp/labels/day'+str(day)+'_indexes.npy')
    newindexes=np.transpose(newindexes)
    
    # Assign variables for storing events
    eventsP = []
    eventsS = []
    eventsN = []
    trace=[]
    newindexes = newindexes*100

    indexP = newindexes[:,0]
    indexS = newindexes[:,1]
    
    # Define window size for segmentation
    start = 7000
    end = 11000
    
    # Loop through and segment all events - obtaining both raw data and labels
    for i in range(len(indexP)):
        windowP = a[int(indexP[i])-start:int(indexP[i])+end,0]
        windowS = a[int(indexP[i])-start:int(indexP[i])+end,1]
        windowNoise = 1 - (windowP + windowS)
        tr_window = b[int(indexP[i])-start:int(indexP[i])+end,:]
        ind = [ind for ind, val in enumerate(windowP) if val == 1]
        
        # Only store windows whcih contain a single event
        if len(ind) == 1:
            eventsN.append(windowNoise)
            eventsP.append(windowP)
            eventsS.append(windowS)
            trace.append(tr_window)
    
    # Store segmented labels    
    labels = [eventsP, eventsS, eventsN]   
    labels = np.array(labels)
    labels = np.swapaxes(labels,0,2)
    labels = np.swapaxes(labels,0,1)
    # Store segmented data
    trace = np.array((trace),dtype=float)
    
    # Save segmented events
    pathlib.Path(output_dir+'/data').mkdir(exist_ok=True, parents=True)
    np.save(output_dir + '/data/Day'+str(day)+'.npy',trace)
    pathlib.Path(output_dir+'/labels').mkdir(exist_ok=True, parents=True)
    np.save(output_dir + 'labels/Day'+str(day)+'.npy',labels)
    print('Saved Day ' +str(day))

