#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create classification vectors where events have been picked in the 
continous streams
@author: jack
"""

""" Lib dependancies """
#==============================================================================
import pathlib
import os 
import obspy as obs
import numpy as np
import re
from obspy.signal.filter import bandpass


""" Miscallaneous functions for data creation"""
#==============================================================================
def gaussian_classvec(x_len, onset, uncert):
    """ Function to calculate a gaussian distribution to incoporate uncertainty
    into manual phase onset (classification vector)."""
    x = np.linspace(1,x_len,x_len)
    gaussian = np.exp(-np.power(x - onset, 2.) / (2 * np.power(uncert, 2.)))
    gaussian[round(onset)]=1                        
    return gaussian

def component_check(traces):
    'Function to match components'
    for tr in traces:
        if tr.stats.channel == 'HHE':
            E = tr
        elif tr.stats.channel == 'HHN':
            N = tr
        elif tr.stats.channel == 'HHZ':
            Z = tr
        else:
            print('Not correct input! channel should\n',
                  + ' be one of:\n HHE\n HHN\n HHZ')
    return E,N,Z  



""" Read in new picks [from continuous stream]"""
#==============================================================================
filesread=[]
output_classdir = '../Data/prep_data2train/temp'

#check output classification data directory exists
pathlib.Path(output_classdir).mkdir(parents=True, exist_ok=True)   
       
#loop through manual picks of each event...
for sdx_root, sdx_dirs, sdx_files in os.walk('../Data/cont_strs/IN22/SDX/metadata/'): 
    for sdx_idx, sdx_file in enumerate(sdx_files):
       
        #loop through raw .mseed data
        for ms_root, ms_dirs, ms_files in os.walk('../Data/cont_strs/IN22/mseed/'):
            for ms_idx, ms_file in enumerate(ms_files):
                
                #find which traces have been manually picked
                sdx_base = sdx_file[:6]
                ms_base = ms_file[:6]
                if ms_base == sdx_base: 
                    print('\n'+"Reading File: " + sdx_base)
                    #print('test... ' + sdx_base + ' -- ' + ms_base)
                    st = obs.read(ms_root + ms_file)
                    
                    sts = st.select(network='C1',station='IN22')
                    df = st[0].stats.sampling_rate
                    
                    try: 
                        E,N,Z = component_check(sts)
                    except:
                        pass
                    
                    # Create a classification vector for storing picks...
                    class_vec = np.zeros((E.stats.npts,3),dtype=float)
                    #De-trend                                                                              
                    E_test = E.data
                    N_test = N.data
                    Z_test = Z.data
                                                                                
                    E_test = E.detrend('linear')
                    N_test = N.detrend('linear')
                    Z_test = Z.detrend('linear')
                    print('BP Filtering...')
                    indexP=[]
                    indexS=[]
                                        
                    #bandpass filter corner freq 2-45Hz (Nyq = 50Hz)
                    E_test = bandpass(E_test, 2, 25, df, corners=4, zerophase=True)
                    N_test = bandpass(N_test, 2, 25, df, corners=4, zerophase=True)
                    Z_test = bandpass(Z_test, 2, 25, df, corners=4, zerophase=True)
                    
                    #match network, station, channels to appropriate picks
                    with open(sdx_root + '/' +sdx_file,'r') as f:
                        line_scan = f.readlines()[4:]
                        
                        #obtain pick information 
                        for line in line_scan:
                            pick_network = line[18:20]
                            pick_station = line[21:26]
                            pick_channel = line[27:31]
                                
                            #remove non alpha-numeric characters...
                            #accounts for different size station names
                            pattern = re.compile('[\W_]+', re.UNICODE)
                            pick_station = pattern.sub('', pick_station)
                            pick_channel = pattern.sub('', pick_channel)
                                
                            pick = line[32:55]
                            pick = pick.strip()
                            pick_yr = int(pick[:4])
                            pick_mth = int(pick[5:7])
                            pick_day = int(pick[8:10])
                            pick_hr = int(pick[11:13])
                            pick_min = int(pick[14:16])
                            pick_sec = float(pick[17:])
                                
                            pick_weight = line[56:60]
                            pick_phase = line[82:84]
                            pick_phase=(re.sub(r'\W+', '', pick_phase))
                            
                            df = E.stats.sampling_rate
                                    
                            man_pick = obs.UTCDateTime(pick_yr, round(pick_mth), pick_day,
                                                           pick_hr, pick_min, pick_sec)
                            onset = man_pick - E.stats.starttime            
                            #create a numpy array for outputting data...

                            # Assign gaussian class vector at manual pick
                            manual_pick = gaussian_classvec(len(E_test), (onset*100), float(40))
                            
                            # Combine all manual picks in the stream
                            if pick_phase == 'P':
                                class_vec[:,0] = class_vec[:,0]+manual_pick
                                indexP.append(onset)
                            elif pick_phase == 'S':
                                class_vec[:,1] = class_vec[:,1]+manual_pick
                                indexS.append(onset)

                            print('Found pick at '+str(onset) + ' --- DAY '
                                  + str(sdx_base[3:]) + '...')     
                            
                            manual_pick=[]                                        
                            #reset output headers
                            P_id = []
                            S_id = []
                        
                        up2=[]
                        # Catch error where there are different # P vs. # S picks
                        if len(indexP) != len(indexS):
                            up2 = min(len(indexS),len(indexP))
                            indexes = [indexP[:up2],indexS[:up2]]
                        else:
                            indexes = [indexP,indexS]
                            indexes = np.array(indexes,dtype=float)
                            
                    # Save new classification vectors 
                    pathlib.Path(output_classdir+'/labels/').mkdir(parents=True, exist_ok=True)   
                    np.save(output_classdir + '/labels/' + str(sdx_base)+'.npy',class_vec)
                    np.save(output_classdir + '/labels/' + str(sdx_base)+'_indexes.npy',indexes)
                    data = np.array((E_test, N_test, Z_test),dtype=float)
                    pathlib.Path(output_classdir+'/data/').mkdir(parents=True, exist_ok=True)   
                    np.save(output_classdir + '/data/' + str(sdx_base)+'.npy',data)
                    
                    indexes=[]       
                    filesread.append(sdx_base)
                    
