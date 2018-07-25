#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to match events with raw data and create classification vectors
@author: Jack
"""
'Lib dependancies'
#==============================================================================
import pickle
import numpy as np
import obspy as obs
import matplotlib.pyplot as plt 
import pathlib
from scipy.signal import butter, lfilter
 
# DEFINE PATH VARIABLES =======================================================
raw_dir = '/home/jack/../../arnas2/MAULE_EVENTS/EVENT'
output_classdir = '/home/jack/../../arnas2/test/'


""" Miscallaneous functions for data creation """
#==============================================================================
def gaussian_classvec(x_len, onset, uncert):
    """ Function to create a gaussian distribution to incoporate uncertainty
    into manual phase onset (classification vector)."""
    x = np.linspace(1,x_len,x_len)
    gaussian = np.exp(-np.power(x - onset, 2.) / (2 * np.power(uncert, 2.)))
    gaussian[round(onset)]=1                        
    return gaussian

def readallstations(rawdata_path):
    'Fn to combine stations containing the same event into a single stream'
    st = obs.read(rawdata_path+'*')
    return st

def setlocalonset(st_starttime, eventheader, onset):
    """Fn to set appropriate onset time for each event, using a combination 
    of: stream start-time, event onset (from header) and local station onset...
    Inputs:
            st_starttime   |    UTCdatetime of st-onset [tr.stats.starttime]
            event_header   |    Event header which contains event onset time
            onset          |    Onset at local station"""
    
    # Get start of event
    evhr = int(eventheader[20:22])
    evmin = int(eventheader[22:24])
    evsec = int(eventheader[24:26])
    # Get start of stream
    sthr = st_starttime.hour
    stmin = st_starttime.minute
    stsec = st_starttime.second
    # Convert all to seconds
    ev_start = (evhr*60*60)+(evmin*60)+evsec
    st_start = (sthr*60*60)+(stmin*60)+stsec
    # Use difference in start time to calculate local onset
    localonset = ev_start-st_start+onset
    return localonset

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

            
           
                
""" Match event metadata with raw data """
#==============================================================================

# Load in event list
with open('./event_lists/nice_format_events.p', 'rb') as handle:
    my_events = pickle.load(handle)
    
# loop through dataset and match events 
miss_count=0    

#Loop through event list
for ev_count, event in enumerate(my_events):
    # get path to raw data from event list
    rawdata_path = raw_dir + event[0]
    # load raw stream
    print('\nReading event  {}\t\t  TOTAL EVENTS = {}'.format(event[0][10:-1], 
                                                        ev_count))
    
    # Try to find raw data, searching over a range of onset times...
    try:
        # First try to match the event header to the raw data
        st = readallstations(rawdata_path)
    except Exception:
        print(' --- Raw data stream not found!... trying 1 second later')       
        try:
            # If stream not found... search for streams within +- 1sec of onset
            plus1sec = str(int(rawdata_path[-3:-1])+1)
            rawdata_path = rawdata_path[:-3] + plus1sec + rawdata_path[-1:]
            st = readallstations(rawdata_path)
        except Exception:
            print(' --- Raw data stream not found!... trying 1 second earlier')       
            try:
                minus1sec = str('0'+str(int(rawdata_path[-3:-1])-2))
                rawdata_path = rawdata_path[:-3] + minus1sec + rawdata_path[-1:]
                st = readallstations(rawdata_path)
            except Exception:
                miss_count = miss_count+1
                print('\x1b[6;31;47m'+'Raw data stream not found!!!'+
                            '\t\t\tMISS COUNT = ' +str(miss_count) 
                            + '\t\t\x1b[0m')
                continue
    
    # Now within the data stream, match stations to appropriate traces 
    for sta in event[1]:
        tr_hits = st.select(station=sta[:4])
                                
        # Match components (ensuring components meet criteria)
        if ( 
            len(tr_hits)==3 and tr_hits[0].stats.delta==0.01
            and len(tr_hits[0])==18000 
            and len(tr_hits[1])==18000
            and len(tr_hits[2])==18000
            ):
        
            # Get onset at each station       
            onset = setlocalonset(tr_hits[0].stats.starttime,
                              event[0],
                              float(sta[7:]))
                
            'Once everything is matched - create classification vector'
            class_vec = gaussian_classvec(tr_hits[0].stats.npts,
                                          onset*(1/tr_hits[0].stats.delta), 40)
            Z = tr_hits.select(component='*Z')
            N = tr_hits.select(component='*N')
            E = tr_hits.select(component='*E')
                                     
            'Also filter raw data to match Iquique processing'
            #De-trend
            Z = Z.detrend('linear')
            N = N.detrend('linear')
            E = E.detrend('linear')
                
            #Bandpass filter
            Z = butter_bandpass_filter(Z[0].data, 2, 25, 1/Z[0].stats.delta)
            N = butter_bandpass_filter(N[0].data, 2, 25, 1/N[0].stats.delta)
            E = butter_bandpass_filter(E[0].data, 2, 25, 1/E[0].stats.delta)
            
            # Store filtered data and classification vector 
            finalclassvec = np.zeros((tr_hits[0].stats.npts,3),dtype=float)
                
            # Re-organise data format to match Iquique 
            data2save = np.array(([Z,N,E]),dtype=float)
            data2save = np.transpose(data2save)
            
            # Save P and S phases seperately
            if sta[4] == 'P':
                # Create full classification vector for P (inc Noise)
                finalclassvec[:,0]=finalclassvec[:,0]+class_vec
                finalclassvec[:,2]=1-class_vec
                
                # Check output directory exists
                pathlib.Path(output_classdir+'/P/'+
                         event[0][10:]).mkdir(parents=True, exist_ok=True)
                # Set ID for saving data
                P_id = tr_hits[0].stats.network+'.'+tr_hits[0].stats.station
                    
                # Combine arrays of class vectors and raw data 
                'This makes it easier to merge the P and S class vectors!'
                matching_arrays = np.concatenate((data2save,finalclassvec),
                                                 axis=1)
                # Save 
                np.savetxt(output_classdir +'/P/'+ event[0][10:] + 
                           P_id + '.csv', matching_arrays, delimiter=',')
        
            elif sta[4] == 'S':
                # Create full classification vector for S (inc Noise)
                finalclassvec[:,1]=finalclassvec[:,1]+class_vec
                finalclassvec[:,2]=1-class_vec

                # Check output directory exists
                pathlib.Path(output_classdir+'/S/'+
                         event[0][10:]).mkdir(parents=True, exist_ok=True)
                # Set ID for saving data
                S_id = tr_hits[0].stats.network+'.'+tr_hits[0].stats.station
                                        
                # Combine arrays of class vectors and raw data 
                'This makes it easier to merge the P and S class vectors!'
                matching_arrays = np.concatenate((data2save,finalclassvec),
                                                 axis=1)
                # Save 
                np.savetxt(output_classdir +'/S/'+ event[0][10:] + 
                           S_id + '.csv', matching_arrays, delimiter=',')
                               
    print('---------- Saved event {} ----------'.format(event[0][10:-1]))
        