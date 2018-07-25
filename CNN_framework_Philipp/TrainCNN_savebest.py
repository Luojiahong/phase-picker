# -*- coding: utf-8 -*-
""" CONVOLUTIONAL NEURAL NETWORK TRAINING 

Train a convolutional neural network on a simulated continuous stream of data.

Events are randomly scaled to simulate differing levels of signal vs. noise
Real noise is appended between events
Window steps along an input continuous trace

J.Woollam --- June 2018
"""
""" Lib dependancies """
#==============================================================================
import os 
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import Input, merge, Conv1D, Conv2D, MaxPooling1D, UpSampling1D, Dropout, Cropping1D, GlobalAveragePooling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from keras import backend as keras
from keras.models import load_model

# Import DeepPhase package for training network
import DeepPhase.gen_cont as gen
import DeepPhase.class_analyse as analyse

""" Define parameters for data generation """
#==============================================================================
# Define parameters for Data Generation 
params = {"shuffle":False,
          "train_data_path":'./Data/final_data2train_maule/data/',
          "label_data_path":'./Data/final_data2train_maule/labels/',
          "windowlen":2800,
          "timestep":1400,
          "batch_size":50} 

# Specify IDs for batches
IDs = ['b{}'.format(i) for i in range(1,45)]
val_IDs = ['b{}'.format(i) for i in range(46,52)]
test_IDs = ['b{}'.format(i) for i in range(53,58)]


# Get generators for data training and validation
training_generator = gen.generate(**params).windowgenerator(IDs)
validation_generator = gen.generate(**params).windowgenerator(val_IDs)
test_generator = gen.generate(**params).windowgenerator(test_IDs)

# Build CNN model
#==============================================================================
""" 2800 SAMPLES ARCHITECTURE """
# Build CNN model
model = Sequential()
inputs=Input((2800,3))
conv1 = Conv1D(12,5, activation='relu', input_shape=(2800,3))(inputs)
print("input shape: (" +str(2800) + ', ' + str(3)+')')
print("\tconv1 shape: ",conv1.shape)
conv1 = Conv1D(24,5, activation='relu',strides=4)(conv1)
print("\tconv1 shape: ",conv1.shape)
conv2 = Conv1D(36, 5, activation='relu',strides=4)(conv1)
print("\t\tconv2 shape: ",conv2.shape)
drop2 = Dropout(0.25)(conv2)
print("\t\tdrop2 shape: ",drop2.shape)
conv3 = Conv1D(48, 5, activation='relu',strides=4)(drop2)
print("\t\t\tconv3 shape: ",conv3.shape)
pool3 = MaxPooling1D(3)(conv3)
print("\t\t\tpool3 shape: ",pool3.shape)
up3 = Conv1D(48,5,activation='relu',padding='same')(UpSampling1D(size=5)(pool3))
print("\t\t\tup3 shape: ",up3.shape)
up4 = Conv1D(24,5,activation = 'relu', padding = 'causal', 
             kernel_initializer = 'he_normal')(UpSampling1D(size=4)(up3))
print("\t\tup4 shape: ",up4.shape)
up5 = Conv1D(12,5,activation = 'relu', padding = 'causal', 
             kernel_initializer = 'he_normal')(UpSampling1D(size =5)(up4))
print("\t\tup5 shape: ",up5.shape)
up6 = Conv1D(3,5,activation = 'softmax', padding = 'causal', 
             kernel_initializer = 'he_normal')(UpSampling1D(size =2)(up5))
print("\tup6 shape: ",up6.shape)

model = Model(input = inputs, output = up6)
model_earlystopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('DeepPhase.hdf5', monitor='loss',verbose=1, save_best_only=True)

#model.compile(optimizer = Adam(lr = 0.01, decay=1e-6), loss = 'categorical_crossentropy', metrics = ['accuracy']) #Original one: 1e-4

model.compile(optimizer = Adam(lr = 1e-4, decay=1e-9), loss = 'categorical_crossentropy', metrics = ['accuracy']) #Original one: 1e-4
print(model.summary())

""" Train CNN model """
#==============================================================================
history = model.fit_generator(generator = training_generator,
                steps_per_epoch = (len(IDs)*params["batch_size"])/params["batch_size"],
                validation_data = validation_generator,
                validation_steps = (len(val_IDs)*params["batch_size"])/params["batch_size"],
                epochs = 30,
                verbose = 1,
                max_q_size=1,
                callbacks=[model_checkpoint, model_earlystopping])

""" Test CNN model """
#==============================================================================
# Generate new continuous stream for testing
store_x_pred, store_y_pred = next(test_generator)

# Plot windows
analyse.window_plotter(store_x_pred,store_y_pred,10)

# Predict model
predictions = model.predict(store_x_pred)      
analyse.prediction_plotter(predictions,store_y_pred, 20)


""" View predictions """
#==============================================================================
'P-PHASE LOSSES' 
# Plot P phase losses [for argmax metric]
fig = plt.figure();
plt.figure(figsize=(7,7));#dpi=400)
plt.subplot(2,1,1);
plt.title('P-phase estimations [using maximum values]');
plt.plot(analyse.loss_function(predictions,store_y_pred,phase='P'),'b.');
plt.xlabel('Sample');#,fontsize=14)
plt.ylabel('Automatic pick error [secs]');#,fontsize=14)
#plt.xlim(0,100)

plt.subplot(2,1,2);
plt.plot(analyse.loss_function(predictions,store_y_pred,phase='P'),'b.');
plt.xlabel('Sample');#,fontsize=14)
plt.ylabel('Automatic pick error [secs]');#,fontsize=14)
plt.ylim([-3, 3]);
#plt.xlim(0,100)
plt.show();
plt.close();


'S-PHASE LOSSES'
# Plot S phase losses [for argmax metric]
fig = plt.figure();
plt.figure(figsize=(7,7));#dpi=400)
plt.subplot(2,1,1);
plt.title('S-phase estimations [using maximum values]');
plt.plot(analyse.loss_function(predictions,store_y_pred,phase='S'),'b.');
plt.xlabel('Sample');#,fontsize=14)
plt.ylabel('Automatic pick error [secs]');#,fontsize=14)
#plt.xlim(100,200)

plt.subplot(2,1,2);
plt.plot(analyse.loss_function(predictions,store_y_pred,phase='S'),'b.');
plt.xlabel('Sample');#,fontsize=14)
plt.ylabel('Automatic pick error [secs]');#,fontsize=14)
plt.ylim([-3, 3]);
#plt.xlim(100,200)
plt.show();
plt.close();

'CNN PICKS VS MANUAL PICKS'
#view autopicks vs. manual picks
fig = plt.figure();
plt.figure(figsize=(22,30));#,dpi=300)
plt.subplot(1,2,1);
plt.plot(store_x_pred[0,:,0],'tab:gray',label='Trace');
#plt.plot(store_y_pred[0,:,0],'r-',label='manual P-pick');
#plt.plot(store_y_pred[0,:,1],'b-',label='manual S-pick');
plt.plot(np.argmax(predictions[0,:,0]),0,'r',label='NeuralNet P-pick',linestyle='--');
plt.plot(np.argmax(predictions[0,:,1]),0,'b',label='NeuralNet S-pick',linestyle='--');
#plt.plot(P_pred[0,:,0],'k',label='NN pick');
plt.title('Neural Network performance',fontsize=18);

# NN P-phase predictions use custom metric, NN S-phase predictions are simply the maximum of the S-class output
for i in range(0,100):
    plt.plot(store_x_pred[i,:,0]/np.amax(store_x_pred[i,:,0])+i,'tab:gray',alpha=0.5);
    plt.plot(predictions[i,:,0]+i,'r-');
    plt.plot(predictions[i,:,1]+i,'b-');
    plt.plot(store_y_pred[i,:,0]+i,'r',alpha=0.3,linestyle='dashed');
    plt.plot(store_y_pred[i,:,1]+i,'b',alpha=0.3,linestyle='dashed');
    #plt.plot(P_pred[i,:,0]+i,'k');
    #plt.vlines(np.argmax(predictions[i,:,0]),0+i/5,1+i/5,'r',linestyles='dashed')
    #plt.vlines(np.argmax(predictions[i,:,1]),0+i/5,1+i/5,'b',linestyles='dashed')

    plt.ylim(-1, 100);
    plt.ylabel('Traces',fontsize=18);
    plt.xlabel('Samples',fontsize=18);
    

legend = plt.legend(frameon = 1, fontsize=14,loc='upper right')
frame = legend.get_frame()
frame.set_color('white')
plt.show()


'HISTOGRAM PLOT'
# plot histogram of losses [for argmax metric]
Plosses=[]
Slosses=[]
for i in range(len(predictions)):
    
    Plosses.append((np.argmax(predictions[i,:,0]) - np.argmax(store_y_pred[i,:,0]))/100)
    Slosses.append((np.argmax(predictions[i,:,1]) - np.argmax(store_y_pred[i,:,1]))/100)

plt.figure(figsize=(15,7));#dpi=400)
plt.subplot(1,2,1);
plt.title('P-phase accuracy',fontsize=16);
plt.hist(Plosses,color='tab:gray',bins=100,edgecolor='black');
plt.xlabel('Automatic pick error [secs]',fontsize=14);
plt.ylabel('Count',fontsize=14)
plt.xlim([-3, 3]);

plt.subplot(1,2,2);
plt.hist(Slosses,color='tab:gray',bins=100,edgecolor='black');
plt.title('S-phase accuracy',fontsize=16)
plt.xlabel('Automatic pick error [secs]',fontsize=14);
plt.ylabel('Count',fontsize=14)
plt.xlim([-3, 3]);
plt.show();
plt.close();

'Print percentage of correct picks considering a specific tolerance in s'
tolerance = 3
P_acc_picks = 0
S_acc_picks = 0

for x in Plosses:
    if abs(x) <= tolerance:
        P_acc_picks+=1
        
for x in Slosses:
    if abs(x) <= tolerance:
        S_acc_picks+=1
        
print(P_acc_picks/len(Plosses) * 100, ' percent of P picks and ' \
      , S_acc_picks/len(Slosses) * 100, 'percent of S picks within ' \
      , tolerance , ' second interval.')

""" Save trained model if its loss is less than that of a previously saved model"""
#==============================================================================
model_name = 'CNN_2800_contstd_xtrapicks_gaussnorm_30_sigma01_savebest'

#load old model history
history_old_model = np.genfromtxt("historyfile.txt", delimiter = ',')
 
#compare loss of old and new model
val_loss_old_model = min(history_old_model)
val_loss_new_model = min(history.history['val_loss'])

# Save model locally if its loss is smaller than the old model's
if val_loss_new_model < val_loss_old_model:
    print('model with lower validation loss found in this run of the script')
    pathlib.Path('./Analyse/models/').mkdir(exist_ok=True, parents=True)
    model.save('./Analyse/models/'+model_name+'.h5')

#save validation loss history
    np.savetxt("historyfile.txt",history.history['val_loss'], delimiter = ',')
