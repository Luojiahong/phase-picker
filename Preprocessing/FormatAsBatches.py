#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to format as data into segmented batches
@author: jack
"""

""" Lib dependancies """
#==============================================================================
import os
import numpy as np
import pathlib
#==============================================================================

def format_as_batches(class_data_dir, output_batchformat_dir, batch_size, x_dim, y_dim,numclasses):
    """Function to format .csv data into batches specified size...
    
    Requires the input dimensions of .csv data where: 
        x_dim = number of samples for each trace
        y_dim = number of training vectors
        numclasses = number of classes
    """
    #define counters and lists for data storage
    batch=[]
    batch_counter=0
    count=0
    IDs=[]
    list_labels=[]
    list_data=[]
    
    #loop through .csv data for storing
    for root, dirs, files in os.walk(class_data_dir):
        for idx, file in enumerate(files):
        
            count = count + 1
            #load into memory in batches of size n and store in a numpy array
            class_data = np.genfromtxt(root + '/'   \
                                       + file, delimiter=',')
        
            #check dimensions of .csv data before inputting into .npy array
            if (np.shape(class_data)[0] == x_dim
                and np.shape(class_data)[1] == y_dim
                ):    
            
                #add data to batch
                batch.append(class_data)
            else: 
                pass 

            #once batch is of speificied size, save as .npy array...
            if len(batch) == batch_size:
                batch_counter = batch_counter + 1
            
                list_labels.append(output_batchformat_dir+str(batch_counter)) 
                list_data.append(output_batchformat_dir+str(batch_counter))    
            
                # X is N windows x M samples, Y is binary classication - format (N, M, 1) 
                batch = np.array(batch, dtype=float)
                X = np.array(batch[:,:,:(y_dim-numclasses)], dtype=float)
                Y = np.array(batch[:,:,(y_dim-numclasses):], dtype=float)
                if len(np.shape(Y)) == 2:
                    Y = Y[:,:,np.newaxis]
            
                #save batches...
                pathlib.Path(output_batchformat_dir+
                             'train/data/').mkdir(parents=True, exist_ok=True)
                pathlib.Path(output_batchformat_dir+
                             'train/labels/').mkdir(parents=True, exist_ok=True)
                np.save(output_batchformat_dir + 'train/data/b'+str(batch_counter),X)
                np.save(output_batchformat_dir + 'train/labels/b'+str(batch_counter),Y)
                print('\ncollecting files...')
                print('Storing batch # ' + str(batch_counter))
                batch=[]
                IDs.append('b'+str(batch_counter))
                break
    return IDs

'Define output variables and create data...'
### ===========================================================================
# Define output variables
class_data_dir = '/home/jack/../../arnas2/MAULE_classvecs/joined/'
output_batchformat_dir = '/home/jack/../../arnas2/MAULE_classvecs/batch_format_maule/'

# format into batches
format_as_batches(class_data_dir, output_batchformat_dir,50,18000,6,3)
