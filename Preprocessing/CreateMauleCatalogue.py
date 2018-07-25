#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to read and reformat cnv events so they can easily be matched to raw 
data...
@author: jack
"""
'Lib dependancies'
#==============================================================================
import pickle


""" Load data """
#==============================================================================
# Read cnv file 
with open('./event_lists/correct_eventlist.cnv') as f:
    content = f.readlines()
# Strip unnecessary formatting    
lines = [line.rstrip('\n') for line in content]


""" Miscallaneous fns for data creation """
#==============================================================================
def slice_fullcnv(fullcnvinfo, slice_at=12):
    'Fn to slice a full cnv info line...'
    
    # Store line info
    ev_info = []
    # Loop through each cnv info line and slice
    for k in range(len(fullcnvinfo)):
        stas = [fullcnvinfo[k][i:i+slice_at] for i in range(0, len(fullcnvinfo[k]),slice_at)]
        ev_info.append(stas)
    # Flatten final nested list into single list containing event info
    total_ev_info = [item for sublist in ev_info for item in sublist]
    return total_ev_info 

def slice_endcnv(endcnvinfo, slice_at=12):
    'Fn to slice an end cnv info line...'
    total_ev_info = [endcnvinfo[i:i+slice_at] 
    for i in range(0, len(endcnvinfo), slice_at)]
    return total_ev_info 

def parse_cnv_event(startline, endline):
    """Fn to parse cnv event info into a nicer format...
    Inputs:
             startline  | headerline index
             endline    | index for end of event info
    Returns:
             headerline | headerline information 
             ev_info    | event information formatted as a list
    """         
    headerline = lines[startline-1]
    fullcnvinfo = lines[startline:endline-2]
    endcnvinfo = lines[endline-2]
    
    # Reformat event data
    tmp_ev1 = slice_fullcnv(fullcnvinfo,12)
    tmp_ev2 = slice_endcnv(endcnvinfo,12)
    ev_info = tmp_ev1+tmp_ev2
    
    return headerline, ev_info

def reformat_hline(hline):
    'Fn to reformat headerlines to match mseed data...'
    yr='20'+hline[:2]
    mth=hline[2:4]
    mth=mth.replace(" ", "0")
    day=hline[4:6]
    day=day.replace(" ", "0")
    hr=hline[7:9]
    hr=hr.replace(" ", "0")
    mns=hline[9:11]
    mns=mns.replace(" ", "0")
    sec=hline[12:14]
    sec=sec.replace(" ", "0")

    mseed_format = str('e'+yr+mth+day+'.'+hr+mns+sec)
    return mseed_format
    

""" Obtain appropriate info and reformat """
#==============================================================================
# Scan through and get index for header-lines/end of event info
hindex = 0
eindex = 0
headerlines=[]
infoend=[]
# Loop througuh each line
for line in lines:
    hindex = hindex + 1
    eindex = eindex + 1 
    if len(line)>0:
        if line[:2] == '10':
            # Store indexes of headerlines
            headerlines.append(hindex)
    elif line[:]=='':
        # Store indexes of end of event info
        infoend.append(eindex)

# Use custom fns to re-order event info into a nicer format and store
my_events = []

for startline, endline in zip(headerlines,infoend):
    hline, ev_info = parse_cnv_event(startline, endline)
    hline = reformat_hline(hline)
    hline = '/'+hline[1:9]+'/'+hline[:]+'/'
    tmp = [hline, ev_info]
    my_events.append(tmp)
    
    # Save event information
    with open('./event_lists/nice_format_events.p', 'wb') as handle:
        pickle.dump(my_events, handle, protocol=pickle.HIGHEST_PROTOCOL)
