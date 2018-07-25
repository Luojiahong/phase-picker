# -*- coding: utf-8 -*-
"""
@author: Jack

Script to merge P and S classification vectors
"""

""" Lib dependancies """
#==============================================================================
import os
import numpy as np
import pathlib
#==============================================================================

def phase_merger(input_dir):
    'Function to merge P & S phases into a single classification vector.'
    P_dir = input_dir + '/P/'
    S_dir = input_dir + '/S/'
    count = 0 
    #loop through P picks of each event...
    for P_root, P_dirs, P_files in os.walk(P_dir): 
        for P_idx, P_file in enumerate(P_files):
            #loop through S picks of each event
            for S_root, S_dirs, S_files in os.walk(S_dir):
                for S_idx, S_file in enumerate(S_files):
                    if P_file == S_file and os.path.basename(P_root) == os.path.basename(S_root):
                       
                        #read phase information
                        a = np.genfromtxt(P_root + '/' + P_file, delimiter=',')
                        b = np.genfromtxt(S_root + '/' + S_file, delimiter=',')
                        
                        #merge classification vector
                        total_classvec = a[:,3:] + b[:,3:]
                        total_classvec[:,2] = total_classvec[:,2]-1
                        #combine with raw data for storing 
                        tostore = np.concatenate((a[:,:3],total_classvec),axis=1)
                        #save 
                        pathlib.Path(input_dir+'/joined/' + os.path.basename(P_root)).mkdir(parents=True, exist_ok=True)
                        np.savetxt(input_dir+'/joined/'+ os.path.basename(P_root)+'/'+P_file,tostore,delimiter=',')
                        print('merged P & S phases for {}:'.format(str(os.path.basename(P_root)+P_file)))
                        count=count+1
                    
            
    print('Total phases paired = {}'.format(count))



### Define input dir ==========================================================
input_dir = '/home/jack/../../arnas2/MAULE_classvecs/joined/'


# merge phases 
phase_merger(input_dir)