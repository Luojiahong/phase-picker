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

#import custom package utility functions for creating SDX pick metadata
import sys
sys.path.append('../')
from DeepPhase import util as u


""" Miscallaneous functions for data creation"""
#==============================================================================
def gaussian_classvec(x_len, onset, uncert):
    """ Function to calculate a gaussian distribution to incoporate uncertainty
    into manual phase onset (classification vector)."""
    x = np.linspace(1,x_len,x_len)
    gaussian = np.exp(-np.power(x - onset, 2.) / (2 * np.power(uncert, 2.)))
    gaussian[round(onset)]=1                        
    return gaussian

""" Read in new picks [from continuous stream]"""
#==============================================================================
filesread=[]
output_classdir = './predictions/manpicks_as_arrays/IN01/'

#check output classification data directory exists
pathlib.Path(output_classdir).mkdir(parents=True, exist_ok=True)   
       
#loop through manual picks of each event...
for sdx_root, sdx_dirs, sdx_files in os.walk('../Data/cont_strs/IN01/SDX/metadata/'): 
    for sdx_idx, sdx_file in enumerate(sdx_files):
       

        print('\n'+"Reading File: " + sdx_file)
        #print('test... ' + sdx_base + ' -- ' + ms_base)
                 
        # Create a classification vector for storing picks...
        class_vec = np.zeros((8640000,3),dtype=float)

        indexP=[]
        indexS=[]
                   
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
                pick_phase = line[85:86]
                pick_phase=(re.sub(r'\W+', '', pick_phase))
                
                man_pick = pick_hr*3600*100 + \
                           pick_min*60*100 + \
                           pick_sec*100
                           
                onset = int(man_pick)            

                # Assign gaussian class vector at manual pick
                manual_pick = np.zeros(8640000)
                manual_pick[onset]=1
                            
                # Combine all manual picks in the stream
                if pick_phase == 'P':
                    class_vec[:,0] = class_vec[:,0]+manual_pick
                    indexP.append(onset)
                elif pick_phase == 'S':
                    class_vec[:,1] = class_vec[:,1]+manual_pick
                    indexS.append(onset)
                    print('Found pick at '+str(onset) + ' --- DAY '
                          + str(sdx_file[3:6]) + '...')     
                            
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
        pathlib.Path(output_classdir).mkdir(parents=True, exist_ok=True)
        # Save new classification vectors        
        np.save(output_classdir + str(sdx_file[:6])+'.npy',class_vec)
        np.save(output_classdir + str(sdx_file[:6])+'_indexes.npy',indexes)
                    
        indexes=[]       
        filesread.append(sdx_file[:6])
                
