#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for manipulating continuous data.
"""

""" Library dependancies """
#==============================================================================
import os
import pathlib

""" Utility functions """
#==============================================================================

def serialize_model(model,model_dir):
    """Fn to serialize a model and save it to disk.
    Inputs     
    
    model:     A Keras model
    model_dir: An output directory to store model files
    """    
    # Make output directory to store model
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_json = model.to_json()
    
    # Serialize model to JSON
    with open(model_dir + '/' + model_dir + ".json", "w") as json_file:
        json_file.write(model_json)
    
    # Serialize weights to HDF5
    model.save_weights(model_dir +'/' + model_dir + ".h5")
    print("Saved model to disk")
    print("Model info stored within local directory: {model_name}/")
    print("Model weights stored as: {model_name}/{model_name}.h5")
    print("Model structure stored as: {model_name}/{model_name}.json")  
    
def create_sdxmetadata(sdx_dir, output_dir):
    """ Function to create SDX pick metadata files.
    
    Inputs: sdx_dir = directory of associated manual sdx picks
            output_dir = define directory to store metadata files on all manual 
            phase picks
    """
    #define list to store SDX information
    instrument = []
    picks = [] 
    phases = []
        
    #segment and store metadata    
    #define SDX files to be read
    for root, dirs, files in os.walk(sdx_dir):
        for idx, file in enumerate(files):
            if file.endswith(".sdx"):
                
                print("Reading File: " + file)
                
                #define list to store SDX information
                instrument = []
                picks = [] 
                phases = []
            
                #scan for pick info
                with open(root + file,"r") as f:
                    searchlines = f.readlines()
                for i, line in enumerate(searchlines):
                    #strip whitespace/end-of-line characters for exact text matching
                    line = line.rstrip()
                    #find pick info
                    if "pick" == line:
                        for l in searchlines[i:i+16]: 
                            #print(l)
                            #assign pick info/instrument info to variables and store
                            instrument_info = searchlines[i+1]
                            pick_info = searchlines[i+2]
                            phase_info = searchlines[i+9:i+13]
                        instrument.append(instrument_info)
                        picks.append(pick_info)
                        phases.append(phase_info)
                        
                        #create a .txt file for each seperate event to store pick info
                        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

                        f = open(output_dir + os.path.splitext(file)[0] + ".txt",'w')
                        #header information...
                        f.write('Data read from correpsonding SDX file:' + '\n')
                        f.write(file + '\n\n')
                        f.write('Instrument/component' + '\t\t\t' + 'Pick information' '\t\t\t' + 'Phase information\n')
                    
                        # print both instrument and pick information to the 
                        # associated event file
                        for item in zip(instrument, picks, phases):
                        
                            #remove preceding whitespace/formatting characters
                            item0 = item[0].rstrip()
                            item1 = item[1].rstrip()
                            item2 = list(map(str.strip, item[2]))
                        
                            #remove associated list formatting
                            item2 = (", ".join( str(e) for e in item2))

                            #print...
                            #format | instrument info | pick info | phase info
                            f.write("%s\t\t%s\t\t%s\n" % (item0,item1,item2))
                       
                        f.close()
