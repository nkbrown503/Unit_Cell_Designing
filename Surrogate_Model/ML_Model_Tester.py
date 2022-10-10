# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 19:19:53 2022

@author: nbrow
"""

from FCC_Surrogate_model import FCC_model
import matplotlib.pyplot as plt 
import numpy as np 
import sys
sys.path.insert(0,r'C:\Users\nbrow\OneDrive - Clemson University\Classwork\Doctorate Research\Python Coding\Unit_Cell_Design\VAE')
from autoencoder import Autoencoder, VAE
import math
import random

surrogate_model= FCC_model() # or create_plain_nest()
_,VAE_Encoder,_,_,_=VAE(20,60,1,24)
VAE_Encoder.load_weights("../VAE/VAE_encoder_24.h5") 

Plot=True
avg_error=[]
It=5
Type='Compression'

if Type=='Tension':
    Norm_Y=np.load('../Result_Files/Normalizing_Tension_Y_Value.npy')
    Norm_X=np.load('../Result_Files/Normalizing_Compression_X_Value.npy')
    surrogate_model.load_weights('checkpoints/UC_Tension_Surrogate_Model_Weights/cp.ckpt')

elif Type=='Compression':
    Norm_Y=np.load('../Result_Files/Normalizing_Compression_Y_Value.npy')
    Norm_X=np.load('../Result_Files/Normalizing_Compression_X_Value.npy')
    surrogate_model.load_weights('checkpoints/UC_Compression_Surrogate_Model_Weights/cp.ckpt')


pred_list=[]
true_list=[]

Colors=['tab:blue','tab:orange','tab:green','tab:red','tab:cyan']
if Type=='Compression':
    UC_All=np.load('Surrogate_Model_UnitCell_C_Data.npy')
    Force_All=np.load('Surrogate_Model_Normalized_Force_C_Data.npy')
else:        
    UC_All=np.load('Surrogate_Model_UnitCell_T_Data.npy')
    Force_All=np.load('Surrogate_Model_Normalized_Force_T_Data.npy')
fig,ax= plt.subplots()

#a=[5883,3193,1193,186,5128,2900,1824,258,3305,3627,1013,916,1268,4350,4323,4276,5332,2105,37,1132,2231,2737,3021,5835,3847,3104,5796,5949,1883,5718,18,3159,633,755,1444,5939,5631,4938,547,2402,5243,5183,393,2946,2128,2004,941,3703,2180,4738,757,2433,93,5835,65,5456,3658,2235,5704,493,2138,1005,109,780,456,2977,4127,3960,4009,3332,5685,4466,4611,3985,4078,2327,5881,2890,5614,4727,4688,5329,5126,5203,5519,3995,4168,3529,4413,4108,2991,4879,4125,1091,2978,353,657,3827,4670,4079,5042,2887,394]
i=0
while i<5:
    if Type=='Compression':
        Val=random.randint(1,6100)
    else:
        Val=random.randint(1,3100)
    #UC=[1111,3001,2972,2345,990]
    
    sys.stdout.write('\rCurrently working on Iteration {}/{}...'.format(i,It))
    sys.stdout.flush()  

    UC=UC_All[Val,:,:,0]
    True_Force=Force_All[Val,:]   
    UC=np.reshape(UC,(1,20,60,1))

    
    ML_input=(VAE_Encoder.predict(UC,verbose=0))


    Pred_Force=surrogate_model.predict(np.reshape(ML_input,(1,24)),verbose=0)[0] 





    if np.min(Pred_Force)>=0 and np.min(True_Force)>=0 and np.max(True_Force)<=1:
        pred_list=np.append(pred_list,Pred_Force)
        true_list=np.append(true_list,True_Force)
    Strain_val=np.linspace(0,1,11)
 
    #Pred_Fit_C=np.array([(Strain**3*Pred_Coef[0])+(Strain**2*Pred_Coef[1])+(Strain*Pred_Coef[2])+Pred_Coef[3] for Strain in Strain_val]) 
    #True_Fit_C=np.array([(Strain**3*True_Coef[0])+(Strain**2*True_Coef[1])+(Strain*True_Coef[2])+True_Coef[3] for Strain in Strain_val]) 

    c1 = np.mean([abs((i-j)/i)*100 for i,j in zip(True_Force[4:],Pred_Force[4:])])
    c2 = np.mean([abs((i-j)/j)*100 for i,j in zip(True_Force[4:],Pred_Force[4:])])
    c = np.mean([abs(i-j) for i,j in zip(True_Force,Pred_Force)])
    #Val=[5574,1036,638,4680,4937,679,4937,938,4771,3939,4599,2272,5367,5999,3865,472,2849,3109,1149,4794,4294,577,70,194,3341,661,877,659,3430,1716,5047,3264,386,5047,126,5088,4020,4273,2725,1990,3263,4877,5755,1847,124,436,4238,4421,5494,3983,713,1410,4135,4416,1613,270,1578,3435,3841,5054,2613,4936,1632,4836,2019,2463,1349,3064,5228,4408,3491,1526,5051,3878,3365,4903,3240,591,2702,2313,1657,1730,2313,2860,3206,5476,4136,263,479,1463,3974,2383,2531,5715,3371,3985,1309,2536,526,2024,832,5958,144,2794,136,1839,5195,5263,4727,5989,785,3066,6094,5847,3823,4211,4193,904,1566,5724,4951,615,4872,3271,5406,4247,2271,4738,456,1124,393,1899,4307,2015,1409,5252,3731,3095,4146,4970,1411,4684,965,203,3732,6059,29,3566,2609,1305,4470,5543,3731,4362,5268,955,649,1914,3054,4837,2548,3594,4264,4909,4463,330,4697,1037,5979,5598,3400,5552,4709,5919,4928,3181,4678,5727,3529,3785,1916,3369,5566,4262,2963]
    if True_Force[-1]>0.1:
        avg_error.append(np.min([c1,c2]))
    if True_Force[-1]>0.9 and True_Force[-1]<1:
        i+=1
        if Plot and np.max(True_Force)==True_Force[-1]:
            fig2,ax2= plt.subplots()
            ax2.imshow(np.reshape(UC,(20,60)),cmap='Blues',origin='lower')
            ax2.axis('off')
            #fig,ax= plt.subplots()
            ax.plot(Strain_val,True_Force,'-',color='{}'.format(Colors[i]),label='True: UC {}'.format(Val))
            #ax.plot(Strain_val,Pred_Force,'--',color='{}'.format(Colors[i]),label='Prediction: UC {}'.format(Val))
            ax.plot(Strain_val,Pred_Force,'--',color='{}'.format(Colors[i]))
            ax.legend(loc='best',prop={'size': 8})
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.set_xlabel('Normalized Displacement')
            ax.set_ylabel('Normalized Force')


print('\nThe average mean % error is: {}%'.format(abs(np.mean(avg_error))))


#plt.scatter(pred_list,true_list)
#plt.xlabel('Real Normalized Force Value')
#plt.ylabel('Predicted Normalized Force Value')

#plt.plot([0,1],[0,1],'r-')


