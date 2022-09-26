# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:14:41 2022

@author: nbrow
"""


import numpy as np
from autoencoder import VAE, Autoencoder
import os
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
import copy
from autoencoder import loss_func, loss_func_auto
import matplotlib.pyplot as plt 


def get_training_data(Use_Samples,Tot_Samples,y_size,x_size):
    x_train=np.zeros((Use_Samples,y_size,x_size))
    
    file_count =Use_Samples
    All_Trials=list(np.linspace(1,Tot_Samples,Tot_Samples).astype('int'))
    Train_Trials=random.sample(All_Trials,int(file_count*.90))
    All_Trials=[x for x in All_Trials if x not in Train_Trials]
    Test_Trials=random.sample(All_Trials, int(file_count*.10))
    x_train=np.zeros((len(Train_Trials),y_size,x_size))
    x_test=np.zeros((len(Test_Trials),y_size,x_size))
    for i in range(0,len(Train_Trials)):
        x_train[i,:,:]=np.load('../ML_Input_Noise_Files/UC_Design_{}.npy'.format(Train_Trials[i]))[0:20,0:60]
    for j in range(0,len(Test_Trials)):
        x_test[j,:,:]=np.load('../ML_Input_Noise_Files/UC_Design_{}.npy'.format(Test_Trials[j]))[0:20,0:60]
    x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)) 
    x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_train.shape[1], x_train.shape[2], 1))
    return x_train,x_test


def train(x_train,x_test,y_size,x_size,num_channels,latent_space_dim, learning_rate,Type):
    if Type=='VAE':
        autoencoder,encoder,decoder, encoder_mu,encoder_log_variance = VAE(y_size,x_size,num_channels,latent_space_dim)
        autoencoder.summary()
        autoencoder.compile(optimizer=Adam(lr=learning_rate), loss=loss_func(encoder_mu, encoder_log_variance))
    elif Type=='Auto':
        autoencoder,encoder,decoder = Autoencoder(y_size,x_size,num_channels,latent_space_dim)
        autoencoder.summary()
        autoencoder.compile(optimizer=Adam(lr=learning_rate), loss=loss_func_auto())

    return autoencoder, encoder,decoder


if __name__ == "__main__":
    x_sample=np.load('../ML_Input_Files/UC_Design_1.npy')[0:20,0:60] #Import Lower Quarter of Unit-Cell
    y_size=np.shape(x_sample)[0]
    x_size=np.shape(x_sample)[1]
    num_channels=1
    Tot_Samples=5000
    Use_Samples=5000
    latent_space_dim=48
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 128
    EPOCHS = 50
    save_name='VAE_training_3-21-22'
    Method='Test'
    Type='VAE' # VAE or Auto
    #x_train,x_test = get_training_data(Use_Samples,Tot_Samples,y_size,x_size)
    if Method=='Train':
        
        x_train=np.load('VAE_Training_Set.npy')
        x_test=np.load('VAE_Testing_Set.npy')
        autoencoder,encoder,decoder = train(x_train,x_test,y_size,x_size,num_channels,latent_space_dim,learning_rate=LEARNING_RATE,Type=Type)
        history=autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, x_test))
        plt.plot(history.history['loss'],label='Training Loss')
        plt.plot(history.history['val_loss'],label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig(save_name+'_LossPlot.png')
        autoencoder.save("model")
        encoder.save("VAE_encoder_40.h5") 
        decoder.save("VAE_decoder_40.h5")
    elif Method=='Test':
        #x_test=np.load('VAE_Testing_Set.npy')
        x_test=get_training_data(5000,Tot_Samples,y_size,x_size)[1]
        #x_test=np.load('AE_Testing_1.npy')
        if Type=='VAE':
            autoencoder,encoder,decoder, encoder_mu,encoder_log_variance = VAE(y_size,x_size,num_channels,latent_space_dim)
            encoder.load_weights("VAE_encoder_48.h5") 
            decoder.load_weights("VAE_decoder_48.h5")
        elif Type=='Auto':
            autoencoder,encoder,decoder = Autoencoder(y_size,x_size,num_channels,latent_space_dim)
            
            encoder.load_weights("AE_encoder_48.h5") 
            decoder.load_weights("AE_decoder_48.h5")
        x_test = np.reshape(x_test, newshape=(500,20,60, 1))
        encoded_data = encoder.predict(x_test)

        #noise_ED=encoded_data+np.random.uniform(low=-.075,high=.075,size=48)
        decoded_data = decoder.predict(encoded_data)
        #noise_dd=decoder.predict(noise_ED)
        hold_dd=copy.deepcopy(decoded_data)
        decoded_data[decoded_data>0.4]=1
        decoded_data[decoded_data<=0.4]=0
        #noise_dd[noise_dd>0.4]=1
        #noise_dd[noise_dd<=0.4]=0
        Error=[]
        for i in range(0,100):
            Num=i
            '''x=np.linspace(1,48,48)
            Num=random.randint(0,0)
            fig_1, axs_1 = plt.subplots()
            fig_2, axs_2 = plt.subplots()
            fig_3, axs_3 = plt.subplots()
            fig_4, axs_4 = plt.subplots()
            fig_5, axs_5 = plt.subplots()
            fig_6, axs_6 = plt.subplots()
            EM=np.zeros((40,120))
            axs_1.imshow(x_test[Num,:,:,0],cmap='Blues',origin='lower')
            axs_1.axis('off')
            axs_2.imshow(decoded_data[Num,:,:,0],cmap='Blues',origin='lower')
            axs_2.axis('off')
            axs_3.imshow(hold_dd[Num,:,:,0],cmap='Blues',origin='lower')
            EM[:20,:60]=x_test[0,:,:,0]
            EM[0:20,60:120]=np.flip(x_test[0,:,:,0],axis=1)
            EM[20:40,0:60]=np.flip(x_test[0,:,:,0],axis=0)
            EM[20:40,60:120]=np.flip(np.flip(x_test[0,:,:,0],axis=0),axis=1)
            axs_4.imshow(EM,cmap='Blues',origin='lower')
            axs_4.axis('off')'''
            Error_Mat=np.sum(abs(x_test[Num,:,:,0]-decoded_data[Num,:,:,0]))
            Error.append(Error_Mat/1200)

            '''axs_5.plot(x,encoded_data[Num],'r-')
            print(encoded_data[Num])
            axs_5.set_xlim(1,48)
            axs_5.set_ylim(-1,1)
            EM=np.zeros((40,120))
            EM[:20,:60]=decoded_data[0,:,:,0]
            EM[0:20,60:120]=np.flip(decoded_data[0,:,:,0],axis=1)
            EM[20:40,0:60]=np.flip(decoded_data[0,:,:,0],axis=0)
            EM[20:40,60:120]=np.flip(np.flip(decoded_data[0,:,:,0],axis=0),axis=1)
            axs_6.imshow(EM,cmap='Blues',origin='lower')
            axs_6.axis('off')

            axs_5.plot(x,encoded_data[Num],'b.',markersize=12)
            axs_5.set_xlabel('Dimension #',fontsize=20)
            axs_5.set_ylabel('Value',fontsize=20)
            axs_5.tick_params(axis='both', which='major', labelsize=20)
            
            
                       
            #axs_4.set_xlabel('Latent Variable')
            #axs_4.set_ylabel('Value')'''
        print('Avg. Error= {}%'.format(round(np.mean(Error),6)*100))