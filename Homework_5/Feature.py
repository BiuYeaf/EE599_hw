import librosa
import os.path as osp
import os
import numpy as np
import h5py
from tqdm import tqdm
import random


English = "F:\\Data\\train\\train_english"
Chinese = "F:\\Data\\train\\train_mandarin"
Hindi   = "F:\\Data\\train\\train_hindi"

M = 1000
ratio = [0.8,0.1,0.1]
threshold = 30

def Shuffledata(X,y):
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(y)


#extract english
eng_dir = os.listdir(English)
eng_feature = np.zeros((1,64))

for file in tqdm(eng_dir):    
    y, sr = librosa.load(osp.join(English, file), sr=16000)
    intervals = librosa.effects.split(y, top_db=threshold)
    y_new = np.zeros((1))
    for interval in intervals:
        y_new = np.concatenate((y_new, y[interval[0]: interval[1]]))
    y = y_new[1:]
    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
    mat = mat.T
    eng_feature=np.concatenate([eng_feature,mat],axis=0)

eng_feature=eng_feature[1:]# total = 163*60000  actual = 8130768
eng_feature=eng_feature[:8130000]
eng_label = 0*np.ones((eng_feature.shape[0],1))

eng_feature=np.reshape(eng_feature,(-1,M,64))
print(eng_feature.shape)
eng_label = np.reshape(eng_label, (-1, M, 1))
print(eng_label.shape)


# extract hindi
hin_dir = os.listdir(Hindi)
hin_feature = np.zeros((1,64))

for file in tqdm(hin_dir):    
    y, sr = librosa.load(osp.join(Hindi, file), sr=16000)
    intervals = librosa.effects.split(y, top_db=threshold)
    y_new = np.zeros((1))
    for interval in intervals:
        y_new = np.concatenate((y_new, y[interval[0]: interval[1]]))
    y = y_new[1:]
    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
    mat = mat.T
    hin_feature=np.concatenate([hin_feature,mat],axis=0)

hin_feature=hin_feature[1:]#total = 41*60000 #actual= 2146331
hin_feature=hin_feature[:2146000]
hin_label = 1*np.ones((hin_feature.shape[0],1))

hin_feature=np.reshape(hin_feature,(-1,M,64))
print(hin_feature.shape)
hin_label = np.reshape(hin_label, (-1, M, 1))
print(hin_label.shape)


#extract chinese
chi_dir = os.listdir(Chinese)
chi_feature = np.zeros((1,64))

for file in tqdm(chi_dir):    
    y, sr = librosa.load(osp.join(Chinese, file), sr=16000)
    intervals = librosa.effects.split(y, top_db=threshold)
    y_new = np.zeros((1))
    for interval in intervals:
        y_new = np.concatenate((y_new, y[interval[0]: interval[1]]))
    y = y_new[1:]
    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
    mat = mat.T
    chi_feature=np.concatenate([chi_feature,mat],axis=0)

chi_feature=chi_feature[1:]#total = 109*60000 actual = 5550619
chi_feature=chi_feature[:5550000]
chi_label = 2*np.ones((chi_feature.shape[0],1))

chi_feature=np.reshape(chi_feature,(-1,M,64))
print(chi_feature.shape)
chi_label = np.reshape(chi_label, (-1, M, 1))
print(chi_label.shape)

eng_train = int(eng_feature.shape[0]*ratio[0])
eng_test  = int(eng_feature.shape[0]*(ratio[0]+ratio[1]))
hin_train = int(hin_feature.shape[0]*ratio[0])
hin_test  = int(hin_feature.shape[0]*(ratio[0]+ratio[1]))
chi_train = int(chi_feature.shape[0]*ratio[0])
chi_test  = int(chi_feature.shape[0]*(ratio[0]+ratio[1]))


train_data = np.concatenate((eng_feature[:eng_train], hin_feature[:hin_train], chi_feature[:chi_train]),axis=0)
train_label= np.concatenate((eng_label[:eng_train],   hin_label[:hin_train],   chi_label[:chi_train]),  axis=0)

val_data   = np.concatenate((eng_feature[eng_train:eng_test], hin_feature[hin_train:hin_test], chi_feature[chi_train:chi_test]),axis=0)
val_label  = np.concatenate((eng_label[eng_train:eng_test],   hin_label[hin_train:hin_test],   chi_label[chi_train:chi_test]),  axis=0)

test_data  = np.concatenate((eng_feature[eng_test:], hin_feature[hin_test:], chi_feature[chi_test:]), axis=0)
test_label = np.concatenate((eng_label[eng_test:],   hin_label[hin_test:],   chi_label[chi_test:]),   axis=0)


random.seed(13579)
Shuffledata(train_data,train_label)
Shuffledata(val_data,val_label)
Shuffledata(test_data,test_label)


f = h5py.File('F:\\Data\\train\\train_set.hdf5', 'w')
f['data'] = train_data
f['label'] = train_label
f.close()

f = h5py.File('F:\\Data\\train\\val_set.hdf5', 'w')
f['data'] = val_data
f['label'] = val_label
f.close()

f = h5py.File('F:\\Data\\train\\test_set.hdf5', 'w')
f['data'] = test_data
f['label'] = test_label
f.close()