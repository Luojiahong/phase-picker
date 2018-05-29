#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create training data for neural network, simulating a continuous str

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
output_dir = '../Data/final_data2train/'

### DEFINE VARIABLES FOR SCALING EVENT AMPLITUDE/FREQUENCY ### 
# mu and sigma to be used in creating a lognormal distribution 
mu = 0.1; sigma=0.5


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


""" Load in Data for constructing noise segments """ 
#==============================================================================

# Load in examples of Noise
st = obs.read('../Data/cont_strs/IN22/raw_data/DG.IN22..HHE.D.2014.120')
trE=st[0]
st = obs.read('../Data/cont_strs/IN22/raw_data/DG.IN22..HHN.D.2014.120')
trN=st[0]
st = obs.read('../Data/cont_strs/IN22/raw_data/DG.IN22..HHZ.D.2014.120')
trZ=st[0]

#merge components into single list
cont_tr = [trZ.data, trN.data, trE.data]
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

# Get segments of noise to be used in 'joining events together'
noise_segments=[]
noise_segments.append(cont_tr[500000:600000])
noise_segments.append(cont_tr[20000:90000])
noise_segments.append(cont_tr[11000:150000])
noise_segments.append(cont_tr[630000:640000])
noise_segments.append(cont_tr[880000:920000])
noise_segments.append(cont_tr[1100000:1160000])
noise_stacked=np.concatenate((noise_segments[0],
                              noise_segments[1],
                              noise_segments[2],
                              noise_segments[3],
                              noise_segments[4],
                              noise_segments[5]),axis=0)

noise_segs = [0,
1,
3,
9,
11,
12,
13,
14,
17,
19,
22,
24,
25,
27,
30,
35,
36,
38,
40,
53,
54,
58,
64,
213,
206,
204,
203,
199,
190,
178,
177,
176,
173,
172,
168,
164,
158,
157]
noise_stacked=np.concatenate(([cont_tr[i*40000:i*40000+40000,:] for i in noise_segs]),axis=0)

# Manually remove any remaining events
noise_stacked2=noise_stacked[:510000,:]
noise_stacked3=noise_stacked[530000:,:]
noise_stacked = np.concatenate((noise_stacked2,noise_stacked3),axis=0)

noise2append=[]
for i in range(82):
    noise2append.append(noise_stacked[i*18000:i*18000+18000])
    
noise2append=np.array(noise2append,dtype=float)


' Load in extra picked events ' 
#==============================================================================
# Load traces
print('Loading segmented traces...')
extra_events_tr = np.empty((1,18000,3),dtype=float)
for root, dirs, files in os.walk('../Data/cont_strs/IN22/segmented_events/data/'):
    for idx, file in enumerate(files): 
        print(root+file)
        a = np.load(root+file)
        extra_events_tr = np.concatenate((extra_events_tr, a), axis=0)
# Load labels        
print('\nLoading segmented labels...')
extra_events_l = np.empty((1,18000,3),dtype=float)
for root, dirs, files in os.walk('../Data/cont_strs/IN22/segmented_events/labels/'):
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
for k in range(1,59):
    #Load in signal data
    tr_data=str('../Data/batch_format_gauss/data/b{}.npy'.format(k))
    label_data=str('../Data/batch_format_gauss/labels/b{}.npy'.format(k))
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
        #Randomly scale events
        appender.append(a[i,:,:]*np.random.lognormal(1,0.4,1))
        label_appender.append(b[i,:,:])
        # Add in noise segments of a random length between phases
        noise_ends = np.random.randint(0,len(noise_stacked)-1000,1)
        noise_len_scaling = np.random.lognormal(mu,sigma,1)
        appender.append(noise_stacked[min(noise_ends):min(noise_ends)+(int(noise_len_scaling[0]*1000))])
        labelfornoise.append(np.zeros((len(noise_stacked[min(noise_ends):min(noise_ends)+(int(noise_len_scaling[0]*1000))]),1)))
        labelfornoise.append(np.zeros((len(noise_stacked[min(noise_ends):min(noise_ends)+(int(noise_len_scaling[0]*1000))]),1)))
        labelfornoise.append(np.ones((len(noise_stacked[min(noise_ends):min(noise_ends)+(int(noise_len_scaling[0]*1000))]),1),dtype=float))
        labelfornoise=np.array(labelfornoise,dtype=float)
        labelfornoise=np.swapaxes(labelfornoise,0,1)
        labelfornoise=np.squeeze(labelfornoise)
        # Remember to also pad labels to match new format
        label_appender.append(labelfornoise)
    
    # Merge new simulated continuous stream
    sim_cont_tr = np.concatenate([appender[i] for i in range(len(appender))],axis=0)
    sim_cont_label = np.concatenate([label_appender[i] for i in range(len(label_appender))],axis=0)
    # Finally add more random noise to match amplitude scaling between noise and signal
    rn = (2*np.random.random_sample(len(sim_cont_tr))-1)*0.2
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


# Plot Scaling Probability distribution
s = np.random.lognormal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, 100, density=True, align='mid')

x = np.linspace(min(bins), max(bins), 10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
       / (x * sigma * np.sqrt(2 * np.pi)))


plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.title('Probability distirbution used to scale events')
plt.show()


