#!/usr/bin/env python3
"""
J.Woollam

Classification data creation programme.

Programme to automatically process catalogues of manually picked data...
Returns the dataset formatted as classification vectors to be used in Neural
Network training:

Requires:
 - Dir of raw trace files (.mseed format)
 - Dir of manual picks (.sdx format)

Outputs:
    Seperate dirs of classification vectors

"""

""" Library dependancies """
#==============================================================================
import pathlib
import os 
import glob
import obspy as obs
import numpy as np
import re
from itertools import islice
from obspy.signal.filter import bandpass
import multiprocessing
from multiprocessing import Process
from shutil import copyfile

""" Define miscellaneous functions for data creation..."""
#==============================================================================
def component_check(traces):
    'Function to match components'
    for tr in traces:
        if tr.stats.channel == 'HHE':
            E = tr
        elif tr.stats.channel == 'HHN':
            N = tr
        elif tr.stats.channel == 'HHZ':
            Z = tr
        else:
            print('Not correct input! channel should\n',
                  + ' be one of:\n HHE\n HHN\n HHZ')
    return E,N,Z

#define a window function 
def window(seq, n):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)        
        yield result

#define a function to create gaussian distributions
def gaussian_classvec(x_len, onset, uncert):
    """ Function to calculate a gaussian distribution to incoporate uncertainty
    into manual phase onset (classification vector)."""
    x = np.linspace(1,x_len,x_len)
    gaussian = np.exp(-np.power(x - onset, 2.) / (2 * np.power(uncert, 2.)))
    gaussian[round(onset)]=1                        
    return gaussian

#define a parallel wrapper function
def runInParallel(fns):
  """parallel wrapper function."""
  proc = []
  for fn in fns:
    p = Process(target=createclassdata,args=(fn,))
    p.start()
    proc.append(p)
  for p in proc:
    p.join()


""" Main functions for creating classification data """
#==============================================================================
#function to read original sdx info creating a metadata file
def create_sdxmetadata(sdx_dir, output_dir):
    """ Function to match .mseed data with manual phase picks.
    
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
                        f = open(output_dir + file[:16] + ".txt",'w')
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
                    
        print('\n' + "Total SDX files read = " + str(idx-1))
        
    
def segment_metadata(meta_dir, output_dir):    
    """ Function to segment metadata in seperate dirs for parallel processing.
    Automatically detects the number of cores and optimises segmentation 
    accordingly"""
    
    #get total number of files
    t_metafiles = len([file for file in glob.iglob(meta_dir+'*', recursive=True)])
    print('\ntoal # of metadata files = ' + str(t_metafiles))
    #get total number of CPU cores
    cores = multiprocessing.cpu_count()
    print('\n# of cores detected = ' + str(cores))
    #obtain segmentation dimensions
    segment_size = round(t_metafiles/cores)
    #final_segment = t_metafiles-(round(t_metafiles/cores)*(cores-1))
    
    #check for/build output dirs
    for i in range(cores):
        pathlib.Path(output_dir+'/meta'+str(i+1)).mkdir(parents=True, exist_ok=True)
        
    #loop metadata and segment into batches for parallel processing
    batch_id=1
    end_seg=0
    seg_count=0
    for root,dirs,files in os.walk(meta_dir):
        for file in files:
            seg_count=seg_count+1
            #assign data to parallel batches
            if seg_count<=segment_size:
                copyfile(root+file,output_dir+'meta'+str(batch_id)+'/'+str(file))
            #final batch accounts for uneven segmentation due to odd #cores
            elif batch_id==cores:
                copyfile(root+file,output_dir+'meta'+str(batch_id)+'/'+str(file))
                end_seg = end_seg+1
            #reset counters and send to new dir when batch_size is met    
            else:
                seg_count=0
                batch_id=batch_id+1
                copyfile(root+file,output_dir+'meta'+str(batch_id)+'/'+str(file))
                
    print('\nSuccesfully segmented metadata into ' + str(batch_id) + ' batches') 
    
            
# main function to create class data 
def createclassdata(meta_dir):
    """ Fn used to create classification data for Neural Net """
    count = 0
    
    #check output classification data directory exists
    pathlib.Path(output_classdir).mkdir(parents=True, exist_ok=True)   
       
    #loop through manual picks of each event...
    for sdx_root, sdx_dirs, sdx_files in os.walk(meta_dir): 
        for sdx_idx, sdx_file in enumerate(sdx_files):
        
            #loop through raw .mseed data
            for ms_root, ms_dirs, ms_files in os.walk(input_mseed_dir):
                for ms_idx, ms_file in enumerate(ms_files):
                
                    #find which traces have been manually picked
                    sdx_base = sdx_file[:16]
                    ms_base = ms_file[:16]
                    if ms_base == sdx_base: 
                        print('\n'+"Reading File: " + sdx_base)
                    
                        #match network, station, channels to appropriate picks
                        with open(sdx_root + sdx_file,'r') as f:
                            line_scan = f.readlines()[4:]
                        
                            #obtain pick information 
                            for line in line_scan:
                                pick_network = line[18:20]
                                pick_station = line[21:26]
                                pick_channel = line[27:31]
                                
                                #remove non alpha-numeric characters...
                                #accounts for different size station names
                                pattern = re.compile('[\W_]+', re.UNICODE)
                                pick_station = pattern.sub('', pick_station)
                                pick_channel = pattern.sub('', pick_channel)
                                
                                pick = line[32:55]
                                pick = pick.strip()
                                pick_yr = int(pick[:4])
                                pick_mth = int(pick[5:7])
                                pick_day = int(pick[8:10])
                                pick_hr = int(pick[11:13])
                                pick_min = int(pick[14:16])
                                pick_sec = float(pick[17:])
                                
                                pick_weight = line[56:60]
                                pick_phase = line[85:87]
                                pick_phase=(re.sub(r'\W+', '', pick_phase))
                                
                                #find appropraite .mseed traces
                                st = obs.read(ms_root + ms_file)
                            
                                sts = st.select(network=pick_network,station=pick_station)
                                
                                try: 
                                    E,N,Z = component_check(sts)
                                except:
                                    pass
                                
                                for tr in st:
                                    if (
                                       pick_network == tr.stats.network 
                                       and pick_station == tr.stats.station    
                                       and pick_channel == tr.stats.channel
                                       and len(tr.data) < 18500
                                       ):
                                        count = count + 1
                                    
                                        df = tr.stats.sampling_rate
                                    
                                        #correlate datetime handles to assign manual
                                        #pick at appropriate point
                                        man_pick = obs.UTCDateTime(pick_yr, round(pick_mth), pick_day,
                                                               pick_hr, pick_min, pick_sec)
                                        onset = man_pick - tr.stats.starttime
            
                                        #De-trend & normalise amplitudes
                                        E_test = E.data[:18000]
                                        N_test = N.data[:18000]
                                        Z_test = Z.data[:18000]
                                        
                                        E_test = E.detrend('linear')
                                        N_test = N.detrend('linear')
                                        Z_test = Z.detrend('linear')
                                        
                                        #bandpass filter corner freq 2-45Hz (Nyq = 50Hz)
                                        E_test = bandpass(E_test, 2, 25, df, corners=4, zerophase=True)
                                        N_test = bandpass(N_test, 2, 25, df, corners=4, zerophase=True)
                                        Z_test = bandpass(Z_test, 2, 25, df, corners=4, zerophase=True)
                                                                               
                                        comp_max = np.max([np.max(E_test),
                                                           np.max(N_test),
                                                           np.max(Z_test)])
    
                                        E_test = np.divide(E_test,comp_max)
                                        N_test = np.divide(N_test,comp_max)
                                        Z_test = np.divide(Z_test,comp_max)
                                                                           
                                        #create a numpy array for outputting data...
                                        output_array = []
                                        output_array = np.array([E_test, N_test, Z_test])
                                        output_array = np.transpose(output_array)
                                        
                                        manual_pick=[]
                                        try:
                                            manual_pick = gaussian_classvec(len(output_array), (onset*df), float(40))
                                        except:
                                            pass
                                        
                                        to_store=[]                                        
                                        to_store = np.empty((len(output_array),np.shape(output_array)[1]+1))
                                        to_store[:,:np.shape(output_array)[1]] = output_array
                                        to_store[:,-1]=manual_pick
                                        output_array = to_store

                                        #reset output headers
                                        P_id = []
                                        S_id = []
                                        
                                        #save P+S phases seperately
                                        if pick_phase == 'P':
                                            pathlib.Path(output_classdir+'/P/'+sdx_base + '/').mkdir(parents=True, exist_ok=True)
                                            P_id = tr.stats.network+'.'+tr.stats.station
                                            try:
                                                np.savetxt(output_classdir +'/P/'+ sdx_base + '/' + P_id + '.csv', output_array, delimiter = ",") 
                                            except:
                                                pass
                                            print(tr)
                                            print(str(count))
                                        elif pick_phase == 'S':
                                            pathlib.Path(output_classdir+'/S/'+sdx_base + '/').mkdir(parents=True, exist_ok=True)
                                            try:                                               
                                                S_id = tr.stats.network+'.'+tr.stats.station
                                            except:
                                                pass                                            
                                            np.savetxt(output_classdir +'/S/' + sdx_base + '/' + S_id + '.csv', output_array, delimiter = ",") 
                                            print(str(count))
                                            print(tr)
                                        else:
                                            break
                                        
                        f.close()
                    

    #print some statistics...                         
    print('total phases matched to .mseed data = ' + str(count))
    
""" Additional functions to reformat Created Classification Data..."""
#==============================================================================
def softmax_classvec(input_dir):
    """Function to create softmax classification vector with categories including:
    Prob(P-phase), Prob(S-phase), Prob(Noise)."""

    P_dir = input_dir + './P/'
    S_dir = input_dir + './S/'
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

                        #obtain P and S phase locations
                        Pphase = np.argmax(a[:,3])
                        Sphase = np.argmax(b[:,3])

                        #create a new gaussian with a more realistic uncert
                        new_gauss1 = gaussian_classvec(len(a),Pphase,float(40))
                        new_gauss2 = gaussian_classvec(len(b),Sphase,float(40))
                        #create softmax class vectors
                        ProbP = new_gauss1
                        ProbS = new_gauss2
                        ProbNoise = 1-(new_gauss1 + new_gauss2)
                        #merge classification vector
                        softmax = np.array([a[:,0],a[:,1],a[:,2],ProbP,ProbS,ProbNoise], dtype=float)
                        softmax = np.transpose(softmax)

                        #save new classification vector
                        pathlib.Path(input_dir+'/softmax/' + os.path.basename(P_root)).mkdir(parents=True, exist_ok=True)
                        np.savetxt(input_dir+'/softmax/'+ os.path.basename(P_root)+'/'+P_file,softmax,delimiter=',')
                        print('Stored softmax class vector for: {}'.format(str(os.path.basename(P_root)+'/'+P_file)))
                        count=count+1


    print('Total phases matched = {}'.format(count))    

def format_as_batches(class_data_dir, output_batchformat_dir, batch_size, x_dim, y_dim, numclasses):
    """Function to format .csv data into batches of specified size...
    
    Requires the input dimensions of .csv data where: 
        x_dim=number of samples for each trace
        y_dim=number of feature vectors
    """
    #define counters and lists for data storage
    batch=[]
    batch_counter=0
    count=0
    IDs=[]
    train_data_paths=[]
    label_data_paths=[]
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
                X = np.array(batch[:,500:,:(y_dim-numclasses)], dtype=float)
                Y = np.array(batch[:,500:,(y_dim-numclasses):], dtype=float)
                if len(np.shape(Y)) == 2:
                    Y = Y[:,:,np.newaxis]
            
            
                #save batches...
                np.save(output_batchformat_dir + 'train/data/b'+str(batch_counter),X)
                np.save(output_batchformat_dir + 'train/labels/b'+str(batch_counter),Y)
                train_data_paths=output_batchformat_dir
                print('\ncollecting files...')
                print('Storing batch # ' + str(batch_counter))
                batch=[]
                IDs.append('b'+str(batch_counter))
                break
    return IDs
    
""" Test for execution as main programme """
#==============================================================================    
    
if __name__ == "__main__":
    print('\n\nExecuting classification data creation programme...\n\n'
          + '\tProgramme runs in parallel + automatically optimises data segmentation for improved performance.\n')
    while True:
        print('Programme requires:\n\n - Directory of MSEED data\n'
              +' - Directory of manual SDX picks\n')
        
        ans = input('\nProceed? [enter y or n]\n')
        if ans == 'y':
            #define required input/output directories 
            input_mseed_dir = input('Enter path to mseed dir...\n')
            sdx_dir = input('Enter path to sdx dir...\n')
            output_classdir = input('Enter path for output classification dir...\n')
            
            #create temporary directories for storing metadata
            meta_dir='./temp/t_meta/'
            pathlib.Path(meta_dir).mkdir(parents=True, exist_ok=True) 
            output_dir='./temp/metabatches/'
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
                
            #create metadata
            print('\n creating metadata for manual sdx picks...')
            create_sdxmetadata(sdx_dir, meta_dir)
            
            #segement metadata
            print('\n\nsegmenting metadata for parallel processing...')
            segment_metadata(meta_dir, output_dir)
            
            #define target directories for parallel processing
            print('\ntarget directories = ' + str(os.listdir(output_dir)))
            parallel_paths = os.listdir(output_dir)
            #get full path to target directories
            parallel_paths = [output_dir + parallel_path +'/' for parallel_path in parallel_paths]
            
            #create classification data
            print('\nRunning in parallel...')
            runInParallel(tuple(parallel_paths))
            
            #ask whether to store optional directories
            ans2 = input('\nDo you want to store the metadata dirs? [enter y or n]\n')
            
            if ans2 == 'y':
                continue
            elif ans2 == 'n':
                os.rmdir('./temp/*')
            else: 
                print('\n!!! Please enter [y] or [n].....\n')
                
            ans3 = input('\nDo you want to create softmax classification vectors? [enter y or n]\n')    
            
            if ans3 == 'y':
                softmax_classvec(output_classdir)
            elif ans3 == 'n':
                continue
            else: 
                print('\n!!! Please enter [y] or [n].....\n')
            os.sys.exit()
        elif ans == 'n':
            break
        else:    
            print('\n!!! Please enter [y] or [n].....\n')
