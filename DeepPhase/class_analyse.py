"""
Functions for analysing DeepPhase Convolutional Neural Network...
"""
""" Library dependancies """
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
#==============================================================================

"""Windowing and plotting functions for data visualisation"""
#==============================================================================

def window_plotter(X_windows, Y_windows, nwindows):
    """function to quickly plot window subsets & manual picks
    Note this plotting function assumes data is trained on multiple classes"""
    count=0
    for idx, temp in enumerate(Y_windows):
        count=count+1
        if count < nwindows:
            #plot windows [normalised for visualisation]
            plt.plot(X_windows[idx,:,0]/np.amax(X_windows[idx,:,0]),'tab:gray', label='E')
            plt.plot(X_windows[idx,:,1]/np.amax(X_windows[idx,:,1]),'tab:gray', label='N')
            plt.plot(X_windows[idx,:,2]/np.amax(X_windows[idx,:,2]),'tab:gray', label='Z')
            #plt.plot(X_windows[idx,:,3], 'b--', label='KT')
            #plt.plot(X_windows[idx,:,4], 'c--', label='Skewness')
            #plt.plot(X_windows[idx,:,5], 'm--', label='STA/LTA')
            
            #overlay classes
            plt.plot(Y_windows[idx,:,0]*np.max(Y_windows[idx,:,0]),'b', label='P-pick')   
            plt.plot(Y_windows[idx,:,1]*np.max(Y_windows[idx,:,1]),'r', label='S-pick')
            plt.plot(Y_windows[idx,:,2]*np.max(Y_windows[idx,:,2]),'k', label='Noise')
            plt.ylim(-1,1)
            plt.legend(loc='lower left')
            plt.show() 
            
            
def prediction_plotter(predictions, Y, nwindows):
    """function to quickly plot prediction results compared to manual picks"""
    count=0
    for idx, temp in enumerate(predictions):
        count=count+1
        if count < nwindows:        
            [plt.plot(predictions[idx,:,i]+i) for i in range(len(np.shape(predictions)))]
            ymin, ymax = plt.ylim()
            [plt.plot((Y[idx,:,i]*np.max(predictions))+i,'tab:gray') for i in range(len(np.shape(predictions)))]
            plt.ylim(ymin,ymax) 
            plt.text(5,0.5,'P')
            plt.text(5,1.5,'S')
            plt.text(5,2.5,'Noise')
            plt.show()
        
               
def loss_function(predictions, class_unitvectors, phase):
    """function to calculate loss/error for each prediction compared to manual 
    phase pick --> error is returned in seconds."""
    loss_secs=[]
    predicted_sample=[]
    manual_pick=[]
    
    #loop through predictions/class_unit_vector and find P phase estimations
    if phase=='P':
        for x, temp in enumerate(predictions):
            if sum(predictions[x,:,0])>0:
                predicted_sample.append(np.argmax(predictions[x,:,0]))
                manual_pick.append(np.argmax(class_unitvectors[x,:,0]))
            else:
                predicted_sample.append(float('nan'))
                manual_pick.append(float('nan'))
    elif phase=='S':            
        for x, temp in enumerate(predictions):
                if sum(predictions[x,:,1])>0:
                    predicted_sample.append(np.argmax(predictions[x,:,1]))
                    manual_pick.append(np.argmax(class_unitvectors[x,:,1]))
                else:
                    predicted_sample.append(float('nan'))
                    manual_pick.append(float('nan'))
    #calculate and return error    
    loss_secs = np.array(predicted_sample) - np.array(manual_pick)
    loss_secs = loss_secs/100

    return loss_secs 


def predict_at_cutoff(predictions, cutoff):
    """function to calculate cut-off value above which phase will be auto-
    matically picked...
    
    filtered_predictions are returned as an array with the same shape as predic
    tions...
    
    The first value which exceeds cut-off threshold is used as P-phase onset.
    
    """
    
    P_onset=[]
    P_only_predictions=np.zeros([int(np.shape(predictions)[0]),
                              int(np.shape(predictions)[1]),
                              int(1)])
    P_only_predictions=np.array(P_only_predictions)
    P_onset=np.array(P_onset)
    
    #predict phase onset if above a certain cut-off threshold
    filtered_predictions=np.zeros([np.shape(predictions)[0],
                                  np.shape(predictions)[1],
                                  1],dtype=float)
    
    for (i,j), element in np.ndenumerate(predictions[:,:,0]):
        if element > cutoff:
            filtered_predictions[i,j,0] = element
            
            
    """define function to identify first nonzero element of a specified 
    dimension of a numpy array..."""        
    def first_nonzero(arr, axis, invalid_val=-1):
        mask = arr!=0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)    
    
    #identify first non-zero element of each row 
    P_onset = first_nonzero(filtered_predictions,axis=1,invalid_val=0)
    P_onset = list(P_onset)
    
    #insert back into appropriate format (same as predictions)
    for row, val in enumerate(P_onset):
        P_only_predictions[row,int(val)] = filtered_predictions[row,int(val),0]    
        
    #set non-zero values equal to 1 for visualisation 
    P_only_predictions[P_only_predictions!=0]=1
    
    return P_only_predictions

def predict_autophase_onset(predictions):
    """ Function to calculate automatic phase predictions, using the max
    value of each classifcation window as the automatically picked phase onset.
    """
    #scan through predictions
    phase_pred=[]
    for i in range(len(predictions)):
        for j in range(len(np.shape(predictions))-1):
            #return phase onsets where there is a 
            phase_pred[i,j] = np.argmax(predictions[i,:,j]) 
    return phase_pred
