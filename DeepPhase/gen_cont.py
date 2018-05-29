#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window generator class for loading data into CNN (for simulated continuous data)."""

""" Library dependancies """
#==============================================================================
import numpy as np

""" Main generator class """
#==============================================================================

class generate(object):
    """Generates batches of data to be read into Neural Net"""
    
    def __init__(self, shuffle, train_data_path, \
                 label_data_path, windowlen, timestep, batch_size=50):
        'Initialisation'
        self.shuffle = shuffle
        self.train_data_path = train_data_path
        self.label_data_path = label_data_path
        self.windowlen = windowlen 
        self.timestep = timestep
        self.batch_size = batch_size # Where batch_size = 
                                     # No of events in a continuous str
    
    def windowgenerator(self, IDs):
        """Generates windows for a batch formatted as a continuous stream"""

        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(IDs)

            # Generate batches
            imax = int(len(indexes))
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [IDs[k] for k in indexes[i:(i+1)]]
                
                # Generate windows for a given input batch
                X, y = self.__get_windows(list_IDs_temp)         
                
                # Shuffle arrays so CNN does not learn an erroneuous temporal order
                X, y = self.__shuffle_in_unison(X,y)
                
                yield X, y
                
            
    def __get_exploration_order(self, IDs):
        """Generates order of exploration"""
        # Find exploration order
        indexes = np.arange(len(IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes            
    
    
    def __get_windows(self, IDs):
        """Generates randomly sampled windows of data for a given batch"""
        # Define variables to store data
        
        for i, ID in enumerate(IDs):
            # Store batch into memory
            X = np.load(self.train_data_path + ID + '.npy')
            # Normalise the batch between 0 and 1 for all components
            #X = X/np.amax(X)
            y = np.load(self.label_data_path + ID + '.npy')        
        
            """Window the current batch, returning windows of specified length
            [windowlen]"""
            X_windows=[]
            y_windows=[]
            
            # Call window_wstep method to window data
            self.timeseries = X
            X_windows = self.__window_wstep()
            Xwindowed_data=[]    
            for window in X_windows:
                X_sd1 = np.std(window[:,0],axis=0)     # JACK HAS MODIFIED SO ALL WINDOWS ARE NORMALISED INDIVUALLY
                X_sd2 = np.std(window[:,1],axis=0)
                X_sd3 = np.std(window[:,2],axis=0)
                
                Xwindowed_data.append(window/np.mean([X_sd1,X_sd2,X_sd3]))       
            Xwindowed_data = np.array(Xwindowed_data)
            
            self.timeseries = y
            y_windows = self.__window_wstep()
            ywindowed_data=[]    
            for window in y_windows:
                ywindowed_data.append(window)       
            ywindowed_data = np.array(ywindowed_data)
            
            return Xwindowed_data, ywindowed_data
        
    
    def __window_wstep(self):
        """ Generate windows of a timeseries, moving along a given 
        timestep.
        Inputs 
        timeseries:     A timeseries data vector
        window_size:    Size of window
        timestep:       Timestep for windowing fn 
        """ 
        #window a given event
        for i in range(0,
                       (int(len(self.timeseries)/self.timestep)-
                          (int(self.windowlen/self.timestep)))*self.timestep,
                        self.timestep):
            window = self.timeseries[i:i+self.windowlen]
            yield window
                
    def __shuffle_in_unison(self, a, b):
        """ Private method to shuffle seperate numpy arrays identically. """
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b
