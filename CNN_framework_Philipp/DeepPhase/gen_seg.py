#@title Data generation class
""" Data generation class for loading data into Neural Net.
    [ For segmented events ] """

""" Library dependancies """
#==============================================================================
import numpy as np
from numpy.random import randint


""" Main generator class """
#==============================================================================

class generate(object):
    """Generates batches of data to be read into Neural Net"""
    
    def __init__(self, x_dim, y_dim, batch_size, shuffle, train_data_path, \
                 label_data_path, windowlen=None, no_windows=None, \
                 centred_ratio=None, softmax=None, numclasses=1, centre_on='S',\
                 timestep=150,n_steps=3):
        'Initialisation'
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_data_path = train_data_path
        self.label_data_path = label_data_path
        self.windowlen = windowlen
        self.no_windows = no_windows
        self.centred_ratio = centred_ratio
        self.softmax = softmax
        self.numclasses=numclasses
        self.centre_on=centre_on
        self.timestep=timestep                  # timestep [defaults to 150]
        
        self.n_steps=n_steps                    # No of steps to move along centred windows about each side from centre 
                                                # [defaults to 3] --> window steps = -nsteps:nsteps
                                                # e.g. for default values - no of windows containing centred phase = 6

        
    def batchgenerator(self, IDs):
        """Generates batches of entire traces (and associated features) to be 
        used in Neural Network"""

        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(IDs)

            # Generate batches
            imax = int(len(indexes))
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [IDs[k] for k in indexes[i:(i+1)]]

                # Generate data
                X, y = self.__data_generation(list_IDs_temp)
                   
                yield X[:,:,:3], y[:,:,:3]
    
    
    def windowgenerator(self, IDs):
        """Generates a specified amount of windows for a given training batch.
        If centred_ratio is not set, defaults to random window samples only. 
        (Centred ratio is an input between 0 and 1 defining the ratio of 
        windows centred on P-phases)"""

        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(IDs)

            # Generate batches
            imax = int(len(indexes))
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [IDs[k] for k in indexes[i:(i+1)]]
                
                                
                if self.centred_ratio == None or self.centred_ratio==0:
                    # Generate randomly windowed data
                    X, y = self.__get_randwindows(list_IDs_temp)
                
                elif self.centred_ratio < 1 and self.centred_ratio > 0:
                    # Generate both types of centred windows
                    X_cP, y_cP = self.__get_centredwindows(list_IDs_temp)
                    X_cS, y_cS = self.__get_centredwindows_other(list_IDs_temp)
                    
                    # Split centred phase windows 50/50  
                    X_cP = X_cP[:int((len(X_cP)/2))]
                    X_cS = X_cS[:int((len(X_cS)/2))]
                    y_cP = y_cP[:int((len(y_cP)/2))]
                    y_cS = y_cS[:int((len(y_cS)/2))]
                    
                    # Merge centred windows
                    X_c = np.concatenate((X_cP,X_cS),axis=0)
                    y_c = np.concatenate((y_cP,y_cS),axis=0)
                    
                    # Get required combination of random vs. centred windows
                    X_r, y_r = self.__get_randwindows(list_IDs_temp)
                    
                    X_r = X_r[:int(round((1-self.centred_ratio)*len(X_r))),:,:]
                    X_c = X_c[:int(round(self.centred_ratio*len(X_c))),:,:]
                    
                    y_r = y_r[:int(round((1-self.centred_ratio)*len(y_r))),:,:]
                    y_c = y_c[:int(round(self.centred_ratio*len(y_c))),:,:]
                    
                    # Join arrays containing both random and centred windows
                    X = np.concatenate((X_r,X_c))
                    y = np.concatenate((y_r,y_c))
                    
                    # Link data in new array to preserve order
                    class_data = np.concatenate((X,y),axis=2)
                    # Randomly shuffle arrays along first axis
                    np.random.shuffle(class_data)
                    # Split data back to input classes [X] vs. class vectors 
                    # [y]
                    X = class_data[:,:,:self.y_dim]
                    y = class_data[:,:,self.y_dim:]
                    
                
                elif self.centred_ratio == 1:
                    # Generate centred windowed data only
                    # !!! Modified to now generate both P and S phases !!!
                    X1, y1 = self.__get_centredwindows(list_IDs_temp)  
                    X2, y2 = self.__get_centredwindows_other(list_IDs_temp)
                    
                    X = np.concatenate((X1,X2),axis=0)
                    y = np.concatenate((y1,y2),axis=0)

                # Check classification vector format
                if self.softmax==True:
                    y=self.__softmax(y)
                    
                # Ensure batches of classification data are of equal size (may be different due to windowing of P-phases)
                spec_size = int(round(np.shape(X)[0],-2))
                                
                yield X[:spec_size,:,:3], y[:spec_size,:,:3]
                
            
    def __get_exploration_order(self, IDs):
        """Generates order of exploration"""
        # Find exploration order
        indexes = np.arange(len(IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes            
    
    
    def __data_generation(self, IDs):
        """Generates data of batch_size samples"""
        # X = (n samples, input tr length, n evaluation params)
        X = np.empty((self.batch_size, self.x_dim, self.y_dim))
        y = np.empty((self.batch_size, self.x_dim, self.numclasses))

        # Generate data
        for i, ID in enumerate(IDs):
            # Store volume
            X[:, :, :] = np.load(self.train_data_path + ID + '.npy')
            y[:,:,:] = np.load(self.label_data_path + ID + '.npy')
            
            # Check classification vector format
            if self.softmax==True:
                y=self.__softmax(y)
                
        return X, y
    
    
    def __get_randwindows(self, IDs):
        """Generates randomly sampled windows of data for a given batch"""
        X = np.empty((self.batch_size, self.x_dim, self.y_dim))
        y = np.empty((self.batch_size, self.x_dim, self.numclasses))

        for i, ID in enumerate(IDs):
            # Store batch into memory
            X[:, :, :] = np.load(self.train_data_path + ID + '.npy')
            y[:,:,:] = np.load(self.label_data_path + ID + '.npy')        
        
            """Randomly sample the current batch, returning a specified number 
            of windows [no_windows] of specified length [windowlen]"""
            X_rand_windows=[]
            y_rand_windows=[]
    
            # Randomly sample data
            while len(X_rand_windows) < self.no_windows:
                # Get random trace centre
                trace_centre = randint((self.windowlen/2), \
                                       (self.x_dim-self.windowlen/2))
                
                # Get random batch sample
                training_sample = randint(0,self.batch_size-1)
                
                # Obtain window around training inputs/classification vector
                windowstart = (trace_centre-self.windowlen/2)
                windowend = (trace_centre+self.windowlen/2)
                Xr_window = X[training_sample,int(windowstart):int(windowend),:]
                yr_window = y[training_sample,int(windowstart):int(windowend),:]
                X_rand_windows.append(Xr_window)
                y_rand_windows.append(yr_window)
    
            # Convert to numpy array 
            X_rand_windows = np.array(X_rand_windows)
            y_rand_windows = np.array(y_rand_windows)
            
        return X_rand_windows, y_rand_windows
    
    
    def __get_centredwindows(self, IDs):
        """Generates windows of data centred on P-phase for a given batch"""
       
        X = np.empty((self.batch_size, self.x_dim, self.y_dim))
        y = np.empty((self.batch_size, self.x_dim, self.numclasses))
        
        for i, ID in enumerate(IDs):
            # Store batch into memory
            X[:, :, :] = np.load(self.train_data_path + ID + '.npy')
            y[:,:,:] = np.load(self.label_data_path + ID + '.npy') 

        """Randomly sample the current batch, returning a specified number 
        of windows [no_windows] of specified length [windowlen]
        """
        X_cen_windows=[]
        y_cen_windows=[]
          
        while len(X_cen_windows) < self.no_windows:
            # Get random trace sample
            training_sample = randint(0,self.batch_size-1)

            # Centre trace on specifed phase [defaults to S-phase]
            if self.centre_on == 'P':
                trace_centre = np.argmax(y[training_sample,:,0])
            elif self.centre_on == 'S':
                trace_centre = np.argmax(y[training_sample,:,1])
            else:
                trace_centre = np.argmax(y[training_sample,:,1])
        
        
            for step in range(-self.n_steps,self.n_steps):
                # Obtain window around training inputs/classification vector
                windowstart = round(trace_centre-self.windowlen/2+self.timestep*step)
                windowend = round(trace_centre+self.windowlen/2+self.timestep*step)

                # Only use phases where entire window lies within trace

                if windowstart > 0: 
                    Xc_window = X[training_sample,int(windowstart):int(windowend),:]
                    yc_window = y[training_sample,int(windowstart):int(windowend),:]

                    # Check to ensure dimensionality of windows are correct
                    if np.shape(Xc_window)[0]==self.windowlen:
                        X_cen_windows.append(Xc_window)
                        y_cen_windows.append(yc_window)
        
        # Convert to numpy array 
        X_cen_windows = np.array(X_cen_windows)
        y_cen_windows = np.array(y_cen_windows)
    
        return X_cen_windows, y_cen_windows

    
    def __get_centredwindows_other(self, IDs):
        """Generates windows of data centred on S-phase for a given batch"""
       
        X = np.empty((self.batch_size, self.x_dim, self.y_dim))
        y = np.empty((self.batch_size, self.x_dim, self.numclasses))
        
        for i, ID in enumerate(IDs):
            # Store batch into memory
            X[:, :, :] = np.load(self.train_data_path + ID + '.npy')
            y[:,:,:] = np.load(self.label_data_path + ID + '.npy') 

        """Randomly sample the current batch, returning a specified number 
        of windows [no_windows] of specified length [windowlen]
        """
        X_cen_windows=[]
        y_cen_windows=[]
          
        while len(X_cen_windows) < self.no_windows:
            # Get random trace sample
            training_sample = randint(0,self.batch_size-1)
            
            #Centre on other-phase
            if self.centre_on == 'P':
                trace_centre = np.argmax(y[training_sample,:,1])
            elif self.centre_on == 'S':
                trace_centre = np.argmax(y[training_sample,:,0])
            else:
                trace_centre = np.argmax(y[training_sample,:,0])
                          
            # Obtain window around training inputs/classification vector
            for step in range(-self.n_steps,self.n_steps):
                windowstart = round(trace_centre-(self.windowlen/2)+(self.timestep*step))
                windowend = round(trace_centre+(self.windowlen/2)+(self.timestep*step))
            
                # Only use phases where entire window lies within trace
                if windowstart > 0: 
                    Xc_window = X[training_sample,int(windowstart):int(windowend),:]
                    yc_window = y[training_sample,int(windowstart):int(windowend),:]
            
                    # Check to ensure dimensionality of windows are correct
                    if np.shape(Xc_window)[0]==self.windowlen:
                        X_cen_windows.append(Xc_window)
                        y_cen_windows.append(yc_window)
        
        # Convert to numpy array 
        X_cen_windows = np.array(X_cen_windows)
        y_cen_windows = np.array(y_cen_windows)
    
        return X_cen_windows, y_cen_windows