#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare manual picks vs autopicks
@author: jack
"""
""" Lib dependancies """
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt


""" Load in arrays of manual picks vs autopicks """
#==============================================================================

manpicks = np.load('./predictions/manpicks_as_arrays/IN01/day120.npy')
autopicks = np.load('./predictions/predictions_as_arrays/CNN2800/DG.IN01..HHE.D.2014.120_preds.npy')
trace = np.load('./predictions/predictions_as_arrays/CNN2800/DG.IN01..HHE.D.2014.120_stream.npy')
manindexesP = [i for i, x in enumerate(manpicks[:,0]) if x == 1]
manindexesS = [i for i, x in enumerate(manpicks[:,1]) if x == 1]

# Apply filter mask to picks
maskP = [autopicks[:,0]>0.001]
cnnPfiltered = autopicks[:,0]*maskP[0]
cnnPfiltered[cnnPfiltered>0] = 0.5
#manpicks[manpicks<0.5] = 0 
            
cnnindexesP = [i for i, x in enumerate(autopicks[:,0]) if x > 0.1]
cnnindexesS = [i for i, x in enumerate(autopicks[:,1]) if x > 0.15]

# Remove multiple recordings of the same phase arrival
singlecnnP=[]
i = 0
previous=0
for _, i in enumerate(cnnindexesP):
    if previous+300>i:
        pass
    else: 
        singlecnnP.append(i)
        previous = i
        
singlecnnS=[]        
i = 0
previous=0        
for _, i in enumerate(cnnindexesS):
    if previous+300>i:
        pass
    else:
        singlecnnS.append(i)
        previous = i

""" Investigate which events are being picked """
#==============================================================================

### Investigate overall ratio of picks ###

# Plot manual vs. autopicks [P-phases]
plt.figure(figsize=(12,4))
plt.title('Manual vs. automatic P-phase picks',fontsize=14)    
for i in manindexesP:
    plt.vlines(i,0.5,1,'tab:gray',alpha=0.6) 
plt.vlines(i,0.5,1,'tab:gray',alpha=0.6,label='Manual picks')    
for i in singlecnnP:
    plt.vlines(i,0,0.5,'g',alpha=0.6)
plt.vlines(i,0,0.5,'g',alpha=0.6,label='CNN picks')
plt.legend(frameon=1,loc='upper right',fontsize=14)   
plt.show()    

# Plot manual vs. autopicks [S-phases]    
plt.figure(figsize=(12,4))
plt.title('Manual vs. automatic S-phase picks',fontsize=14)    
for i in manindexesS:
    plt.vlines(i,0.5,1,'tab:gray',alpha=0.6) 
plt.vlines(i,0.5,1,'tab:gray',alpha=0.6,label='Manual picks')    
for i in singlecnnS:
    plt.vlines(i,0,0.5,'g',alpha=0.6)
plt.vlines(i,0,0.5,'g',alpha=0.6,label='CNN picks') 
plt.legend(frameon=1,loc='upper right',fontsize=14)
plt.show()    

auto2man_ratioP = round(len(singlecnnP)/len(manindexesP),3)
auto2man_ratioS = round(len(singlecnnS)/len(manindexesS),3)

print('\nRatio of manual vs. CNN P-Phase picks --- {}'.format(auto2man_ratioP))
print('\nRatio of manual vs. CNN S-Phase picks --- {}'.format(auto2man_ratioS))

### Investigate indiviual events ### 

# Loop through and view where events are being predicted
for i in range(0,8640000-100000,100000):
    
    start = i
    end = i+100000

    # Only select picks within plot window
    plot_mindexesP = [ x for x in manindexesP if x < end and x > start]
    plot_mindexesS = [ x for x in manindexesS if x < end and x > start]
    plot_aindexesP = [ x for x in cnnindexesP if x < end and x > start]
    seg_singlecnnP = [ x for x in singlecnnP if x < end and x > start]
    seg_singlecnnS = [ x for x in singlecnnS if x < end and x > start]
    # Plot manual vs. autopicks 
    plt.figure(figsize=(12,6))    
    plt.vlines(plot_mindexesP,-1,0,'g',linestyles='dashed')#manual p picks 
    plt.vlines(plot_mindexesS,-1,0,'b',linestyles='dashed')#manual s picks
    #plt.vlines(plot_aindexesP,-1,0,'g',linestyles='solid',alpha=0.5) # ?automatic p picks
    
    # Overlay normalised trace and original cnn predictions    
    plt.plot(np.linspace(start,end,100000),
             trace[start:end,0]/np.amax(trace[start:end,0]),'tab:gray',alpha=0.5)# grey trace    
    plt.plot(np.linspace(start,end,100000),autopicks[start:end,1],'r',alpha=0.5)# s onset probabilities
    plt.plot(np.linspace(start,end,100000),autopicks[start:end,0],'k',alpha=0.5)# p onset probabilities
    #plt.vlines(seg_singlecnnP,-1,1,'g',alpha=0.5)# p picks of the cnn if there is a certain distance to cnn s and p picks  
    #plt.vlines(seg_singlecnnS,0,1,'b',alpha=0.5)#  s picks of the cnn if there is a certain distance to cnn s and p picks
    plt.show()    
    
    """ Functions and to accelerate any prediction analysis"""
#==============================================================================    
    
### Miscellaneous fns to aid analysis #####

def indexes2preds(onsets, pred_len=8640000):
    """Fn to convert array of onset indexes to arrays of predictions"""
    preds = np.zeros(pred_len)
    for i in onsets:
        preds[i]=1
    return preds

def parsePpicks(P_preds):
    """ Fn to sort P-pick predictions, using the first value past cutoff as the
    index for each P-phase pick."""
    
    def consecutive(data, stepsize=1):
        'Fn to group consecutive indexes'
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    
    # Group P-pick predictions
    P_gr = consecutive(np.nonzero(P_preds)[0],stepsize=1)
    # Get first index for each group
    Pidx = [P_gr[i][0] for i, _ in enumerate(P_gr)]
    # Remove onsets located suspisciously close to each other
    P_onsets = [i for i in enumerate(Pidx) if abs(Pidx[i[0]]-Pidx[i[0]-1])>200]
    P_onsets = [x[1] for x in P_onsets]
    return P_onsets


### Picker Fns ####
    
def Ppicker(Prob_phases, P_cutoff, opt_plot=False):
    """ Fn to predict P-phase onsets for a continous stream.
    Inputs:
        Prob_phases          | CNN output probabilties for the continuous str
        P_cutoff             | Cutoff threshold above which P-phase is 
                               predicted
        [opt_plot]           | Optional plot argument
    Outputs:
        P_preds              | Boolean array of P-phase onsets
    """
    
    # Predict P-phase onsets using defined cut-off
    maskP = [Prob_phases[:,0]>P_cutoff]
    P_preds = Prob_phases[:,0]*maskP[0]
    P_preds[P_preds!=0]=1
    
    # Optional plot
    if opt_plot==True:        
        for i in range(0,len(autopicks)-100000,100000):
        
            start = i
            end = i+100000
            
            plt.figure(figsize=(12,4))
            plt.plot(trace[start:end,0]/np.amax(trace[start:end,0]),'tab:gray',alpha=0.7); 
            plt.plot(P_preds[start:end],'g--',alpha=0.4,label='P-pick');
            plt.plot(Prob_phases[start:end,0],'r',alpha=0.4,label='Prob(P)');
            plt.plot(Prob_phases[start:end,1],'b',alpha=0.4, label='Prob(S)'); 
            plt.legend(loc=1)
            plt.show()
            
    return P_preds

def event_picker(autopicks,P_cut,S_sum,S_start,S_end,opt_plot=False):
    """Fn to to filter predictions based on the temporal relationship between P 
    and S phases.
    
    P-onsets are defined by a given cutoff threshold (P_cut), S phases are then 
    identified in a window behind the P-phase onset [S_start:S_end]. If the
    S-phase probabilities within this window meet a defined cut-off (S_sum),
    then a S-phase is determined to be present with the onset being the maximum
    probability within [S_start:S_end] window.
    
    Inputs:
            P_cut      | Prob(P) cutoff for P-phase picks
            S_sum      | S_sum cutoff for S-phase picks
            S_start    | Start of window to search for S-phases
            S_end      | End of window to search for S-phases
            [opt_plot] | Optional plot argument (defaults to False)
    Returns:
            events     | List of segmented events
    """
    #store events; and define optional event segmentation params
    events=[];   ev_st=-3000; ev_end=+5000    
    # Predict P-phase onsets using defined cut-off
    maskP = [autopicks[:,0]>P_cut]
    P_preds = autopicks[:,0]*maskP[0]
    P_preds[P_preds!=0]=1    
    
    # Filter predictions to get a single index associated with each pick
    P_onsets = parsePpicks(P_preds)

    # Convert onset indexes to array of predictions
    P_preds = indexes2preds(P_onsets)
    
    #Check P(S-phase) behind P(P-phase) filtering false picks...
    for i in P_onsets:
        S2check = autopicks[i+S_start:i+S_end,1] 
        if sum(S2check) > S_sum:
            events.append(autopicks[i+ev_st:i+S_end+ev_end,:])
            
            # Optional plot routine
            if opt_plot == True:
                plt.plot(trace[i+ev_st:i+S_end+ev_end,0]/
                         np.amax(trace[i+ev_st:i+S_end+ev_end,0]),
                         'tab:gray',
                         label='Trace',
                         alpha=0.4)
                plt.plot(autopicks[i+ev_st:i+S_end+ev_end,0],'r',
                         label='Prob(P)',
                         alpha=0.4)
                plt.plot(autopicks[i+ev_st:i+S_end+ev_end,1],'b',
                         label='Prob(S)',
                         alpha=0.4)

                plt.vlines(abs(ev_st),-1,1,'r',label='P-pick')
                plt.vlines(abs(ev_st)+np.argmax(S2check)+S_start,-1,1,'b',
                           label='S-pick')
                
                plt.legend(loc=1)
                plt.show()

    return events
    
    

        