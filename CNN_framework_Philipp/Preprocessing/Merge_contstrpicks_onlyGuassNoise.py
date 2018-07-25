#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create training data for neural network, simulating a continuous str
Only using Gaussian noise to pad inbetween events...

Creates events sampling from initial Iquique catalogue + from additonal 
continous str picks... 

@author: jwol
"""

""" Lib dependancies """
#==============================================================================
import numpy as np
from matplotlib import pyplot as plt
import obspy as obs
from scipy.signal import butter, lfilter
import pathlib
import os

#### DEFINE OUTPUT DIR ####
output_dir = '../Data/final_data2train_maule/'

### DEFINE VARIABLES FOR SCALING EVENT AMPLITUDE ### 
# mu and sigma to be used in creating a lognormal distribution 
mu = 0; sigma=0.25
### DEFINE VARIABLES FOR SCALING EVENT FREQUENCY### 
# mu and sigma to be used in creating a lognormal distribution 
mu2 = -0; sigma2 = 0.3

""" Miscellaneous Fns for manipulating data """
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


def conv2readable(samples):
    """ Fn to convert #samples to equivalent time in hours/mins/secs """
    secs = samples/100
    mins = secs/60
    hrs = mins/60
    
    print('\n--- '+str(int(samples))+' samples\n--- '
          +str(int(secs))+' secs\n--- '
          +str(round(mins,2))+' mins\n--- '
          +str(round(hrs,2))+' hours')

def shuffle_in_unison(a, b):
    """ Private method to shuffle seperate numpy arrays identically. """
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def signal_taper(timeseries, perc2taper, optplot=False):
    """Fn to taper ends of a timeseries, simple trapezoid taper fn is created.
    Inputs |  timeseries : Input timeseries
           |  perc2taper : % of ends to apply taper
           |  optplot :  Optional plot [defualts to False]
    """
             
    # Get length of timeseries
    tl = len(timeseries)
    # Taper ends according to specified input
    ends = int((tl/100)*perc2taper)
    # Perform linear interpolation for taper edges
    xp = [0,ends]
    fp = [0,1]
    x = np.linspace(0,ends,ends+1)    
    # Taper start and end of timeseries
    start_taper = np.interp(x,xp,fp)
    start_taper = np.array((start_taper, start_taper, start_taper))
    end_taper = np.fliplr(start_taper)
    end_taper = np.transpose(end_taper)
    start_taper = np.transpose(start_taper)
    # Create final taper window
    taper_window = np.concatenate((start_taper,np.ones((tl-(2*ends)-2,3),
                                                       dtype=float),end_taper))
    
    # Optional plot
    if optplot == True:
        plt.plot(taper_window);
        plt.show();
 
    return taper_window    


' Load in extra picked events ' 
#==============================================================================
# Load traces
print('Loading segmented traces...')
extra_events_tr = np.empty((1,18000,3),dtype=float)
for root, dirs, files in os.walk('../Data/prep_data2train/temp/segmented_events/data/'):
    for idx, file in enumerate(files): 
        print(root+file)
        a = np.load(root+file)
        extra_events_tr = np.concatenate((extra_events_tr, a), axis=0)
# Load labels        
print('\nLoading segmented labels...')
extra_events_l = np.empty((1,18000,3),dtype=float)
for root, dirs, files in os.walk('../Data/prep_data2train/temp/segmented_events/labels/'):
    for idx, file in enumerate(files): 
        print(root+file)
        a = np.load(root+file)
        extra_events_l = np.concatenate((extra_events_l, a), axis=0)
extra_events_l=extra_events_l[1:]
extra_events_tr=extra_events_tr[1:]

del a
# Evenly distribute new segmented events within batches
n_new = round(len(extra_events_tr)/58)-1


""" Load in Signal Data and reconstruct simulating a continuous stream """ 
#==============================================================================
# Loop through batches of data
for k in range(1,94):
    #Load in signal data
    tr_data=str('../../../myData/batch_format_maule/train/data/b{}.npy'.format(k))
    label_data=str('../../../myData/batch_format_maule/train/labels/b{}.npy'.format(k))
    a = np.load(tr_data)
    b = np.load(label_data)
    
    # ADD NEWLY PICKED EVENTS
    a = np.concatenate((a,extra_events_tr[k*n_new:(k*n_new)+n_new]))
    b = np.concatenate((b,extra_events_l[k*n_new:(k*n_new)+n_new]))

    # Shuffle the new events - shuffling both data and label arrays identically
    a, b = shuffle_in_unison(a,b)
    
    # Create a new stream of data which attempts to simulate continuous data 
    appender=[]
    label_appender=[]
    
    # Loop through events
    for i in range(len(a)):
        labelfornoise=[]
        #Randomly scale events + taper window edges
        taper_window = signal_taper(a[i,:,:],5)
        appender.append(a[i,:,:]*np.random.lognormal(mu,sigma,1)*taper_window)
        label_appender.append(b[i,:,:])
        # Add in noise segments of a random length between phases
        noise_ends = np.random.randint(0,150,1)
        noise_len_scaling = np.random.lognormal(mu2,sigma2,1)
        #noise_len_scaling = [1.5]

        appender.append(np.zeros((noise_ends[0]*int(noise_len_scaling[0]*100),3)))
        labelfornoise.append(np.zeros((noise_ends[0]*int(noise_len_scaling[0]*100),1)))
        labelfornoise.append(np.zeros((noise_ends[0]*int(noise_len_scaling[0]*100),1)))
        labelfornoise.append(np.ones((noise_ends[0]*int(noise_len_scaling[0]*100),1)))
        labelfornoise=np.array(labelfornoise,dtype=float)
        labelfornoise=np.swapaxes(labelfornoise,0,1)
        labelfornoise=np.squeeze(labelfornoise)
        # Remember to also pad labels to match new format
        label_appender.append(labelfornoise)
    
    # Merge new simulated continuous stream
    sim_cont_tr = np.concatenate([appender[i] for i in range(len(appender))],axis=0)
    sim_cont_label = np.concatenate([label_appender[i] for i in range(len(label_appender))],axis=0)
    # Finally add more random noise to match amplitude scaling between noise and signal
    rn = (2*np.random.random_sample(len(sim_cont_tr))-1)*150
    rn = np.array((rn,rn,rn),dtype=float)
    rn = np.transpose(rn)
    
    sim_cont_tr = sim_cont_tr+rn
    # Optional plot
    #plt.plot(sim_cont_tr[:,0])
    #plt.plot(sim_cont_label)
    #plt.show()

    # Save new numpy arrays
    pathlib.Path(output_dir + 'data/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_dir + 'labels/').mkdir(parents=True, exist_ok=True)

    save_train=str(output_dir + 'data/b{}.npy'.format(k))
    save_label=str(output_dir + 'labels/b{}.npy'.format(k))
    
    np.save(save_train,sim_cont_tr)
    np.save(save_label,sim_cont_label)
    print('Reformatted batch {} to simulate continuous data...'.format(k))

#==============================================================================

### Plot Scaling Probability distribution ###
s = np.random.lognormal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, 100, density=True, align='mid')

# Add a PDF
x = np.linspace(min(bins), max(bins), 10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
       / (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')

# Label
plt.axis('tight')
plt.title('Probability distribution used to scale events')
plt.xlabel('Amp scaling factor')
plt.ylabel('count')
plt.show()


### Plot event frequency probability distribution ###
s = np.random.lognormal(mu2, sigma2, 1000)
count, bins, ignored = plt.hist(s, 100, density=True, align='mid')

# Add a PDF
x = np.linspace(min(bins), max(bins), 10000)
pdf = (np.exp(-(np.log(x) - mu2)**2 / (2 * sigma2**2))
       / (x * sigma2 * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')

# Label
plt.axis('tight')
plt.title('Probability distribution used to scale event frequency')
plt.xlabel('Event frequency scaling factor')
plt.ylabel('count')
plt.show()