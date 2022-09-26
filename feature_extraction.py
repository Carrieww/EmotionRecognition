import os
import pandas as pd
from zipfile import ZipFile
from sklearn.model_selection import StratifiedShuffleSplit
from random import shuffle
import numpy as np
import librosa
import glob
import pickle
import joblib

from Extract_Features_trash import extract_features
from utils.util import mkdirs

def load_features(config, file_name: str):
    """
    Load features from "{config.feature_folder}/*.p"
    Args:
        config:
        train (bool):
    Returns:
        - X ([np.ndarray]): features
        - Y ([np.ndarray]): labels
    """
    feature_path = os.path.join(config.feature_folder, file_name)

    features = pd.DataFrame(
        data = joblib.load(feature_path),
        columns = ['features','spec', 'emotion']
    )

    X1 = list(features['features'])
    X2 = list(features['spec'])
    Y = list(features['emotion'])
    Y = np.array(Y)
    return X1,X2, Y


def get_data_path(config):
    # dataset_list=glob.glob(r'./Dataset/*.zip')
    # Zip_file_path = dataset_list[2] # CREMA

    with ZipFile(config.Zip_file_path, 'r') as zip:
        Name_list = np.array(zip.namelist())
        files = Name_list[[x.endswith(".wav") for x in Name_list]]
        
    if config.data_size != 'None':
        files=files[:config.data_size]
    print(files.shape)
    shuffle(files) # file names are randomly shuffled every time
    return files

def get_min_max(config,files:list):

    min_,max_ = 100,0
    for file in files:
        with ZipFile(config.Zip_file_path, 'r').open(file) as f:
            sound_file,sr=librosa.load(f,sr=config.sample_rate)
            t = sound_file.shape[0]/sr
            if t < min_:
                min_ = t
            elif t > max_:
                max_ = t
                max_sf=file
    # min_,max_,max_sf=get_min_max(files)
    print(min_)
    print(max_)
    print(max_sf)
    return min_,max_

def features(X,sr:float,FRAME_SIZE:int,HOP_LENGTH:int):
    stft = np.abs(librosa.stft(X))

    # fmin and fmax correspond the max and min frequency of human voices
    pitches, magnitudes = librosa.piptrack(y=X, sr=sr, S=stft, fmin=70, fmax=400,n_fft=FRAME_SIZE,hop_length=HOP_LENGTH)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)

    # spectral_centroid
    cent = librosa.feature.spectral_centroid(y=X, sr=sr,n_fft=FRAME_SIZE,hop_length=HOP_LENGTH)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    # spectral_flatness
    flatness = np.mean(librosa.feature.spectral_flatness(y=X,n_fft=FRAME_SIZE,hop_length=HOP_LENGTH))

    # MFCC
    mfcc = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=24,n_fft=FRAME_SIZE,hop_length=HOP_LENGTH)
    mfccs = np.mean(mfcc.T, axis=0)
    mfccsstd = np.std(mfcc.T, axis=0)
    mfccmax = np.max(mfcc.T, axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    # melspectrogram
    mel_spec=librosa.feature.melspectrogram(y=X, sr=sr,n_fft=FRAME_SIZE,win_length = 512,hop_length=HOP_LENGTH,n_mels=128,fmax=sr/2)
    mel_mean = np.mean(mel_spec.T, axis=0)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # spectral contrast, ottava
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr,n_fft=FRAME_SIZE,hop_length=HOP_LENGTH).T, axis=0)

    # zero crossing rate
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    S, _ = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    # rmse
    rmse = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel_mean, contrast))

    return ext_features, mel_spec_db

def addAWGN(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30):
    signal_len = len(signal)
    # Generate White Gaussian noise
    noise = np.random.normal(size=(augmented_num, signal_len))
    # Normalize signal and noise
    norm_constant = 2.0**(num_bits-1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(snr_low, snr_high)
    # Compute K (covariance matrix) for each noise 
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K  
    # Generate noisy signal
    return signal + K.T * noise

def extract_features(config,file,max_,trainset):
    with ZipFile(config.Zip_file_path, 'r').open(file) as f:
        X,sr=librosa.load(f,sr=config.sample_rate)
    max_x = X.shape[0] / sr
    if max_x < max_:
        length = (max_ * sr) - X.shape[0]
        X = np.pad(X, (0, int(length)), 'constant')
    if trainset == True:
        augmented_signals = addAWGN(X)
        ext_features=[]
        ext_spec = []
        for i in range(augmented_signals.shape[0]):
            ext_features_aug,ext_spec_aug = features(augmented_signals[i], sr,1024,256)
            ext_features.append(ext_features_aug)
            ext_spec.append(ext_spec_aug)
        ext_features_ori,ext_spec_ori = features(X, sr,1024,256)
        ext_features.append(ext_features_ori)
        ext_spec.append(ext_spec_ori)
    else:
        ext_features,ext_spec = features(X, sr,1024,256)
        ext_features = [ext_features]
        ext_spec = [ext_spec]
    return ext_features,ext_spec

def split_data(config,files):
    Y = []
    new_files=[]
    for file in files: # only training samples need to be augmented
        file_name = os.path.basename(file)
        if config.dataset=="Crema":
            label = file_name.split('_')[2] 
            label = config.Emotions_map[file_name.split('_')[2]]
        elif config.dataset=="Ravdess":
            label = config.Emotions_map[file_name.split('-')[2]]
        elif config.dataset=="Savee":
            if file_name.split('_')[1][1].isnumeric():
                label = config.Emotions_map[file_name.split('_')[1][0]]
            else:
                label = config.Emotions_map[file_name.split('_')[1][:2]]
        if label in config.class_labels:
            Y.append(label)
            new_files.append(file)
    new_files=np.array(new_files)
    sss = StratifiedShuffleSplit(test_size=config.test_ratio, random_state=config.seed)
    for train_index, test_index in sss.split(new_files,Y):
        train_set, test_set = new_files[train_index], new_files[test_index]
        # Y_train, Y_test = Y[train_index], Y[test_index]
    
    val_set = test_set
    return train_set, val_set, test_set

def get_features(config, train):
    if train == True:
        files = get_data_path(config)
        # train test split
        train_set, val_set, test_set = split_data(config,files)

        _,max_ = get_min_max(config,files)

        # get X_train and Y_train, with augmentation
        X1_train,X2_train,Y_train=[],[],[]
        train_feature = []
        for file in train_set: # only training samples need to be augmented
            file_name = os.path.basename(file)
            if config.dataset=="Crema":
                label = file_name.split('_')[2] 
                label = config.Emotions_map[file_name.split('_')[2]]
            elif config.dataset=="Ravdess":
                label = config.Emotions_map[file_name.split('-')[2]]
            elif config.dataset=="Savee":
                if file_name.split('_')[1][1].isnumeric():
                    label = config.Emotions_map[file_name.split('_')[1][0]]
                else:
                    label = config.Emotions_map[file_name.split('_')[1][:2]]
            print(label)
            if label in config.class_labels:
                # Y_train.append(label)
                features,spec = extract_features(config, file, max_,trainset=True) # features is a list of three lists
                for i in range(len(features)):
                    train_feature.append([features[i],spec[i],config.class_labels.index(label)])
                    X1_train.append(features[i])#[file, i, config.class_labels.index(label)])
                    X2_train.append(spec[i])#[file, i, config.class_labels.index(label)])
                    Y_train.append(config.class_labels.index(label))


        X1_val,X2_val,Y_val=[],[],[]
        val_feature = []
        for file in val_set: # only training samples need to be augmented
            file_name = os.path.basename(file)
            if config.dataset=="Crema":
                label = file_name.split('_')[2] 
                label = config.Emotions_map[file_name.split('_')[2]]
            elif config.dataset=="Ravdess":
                label = config.Emotions_map[file_name.split('-')[2]]
            elif config.dataset=="Savee":
                if file_name.split('_')[1][1].isnumeric():
                    label = config.Emotions_map[file_name.split('_')[1][0]]
                else:
                    label = config.Emotions_map[file_name.split('_')[1][:2]]
            print(label)
            if label in config.class_labels:
                features,spec = extract_features(config, file, max_,trainset=False) # features is a list of three lists
                val_feature.append([features[0],spec[0],config.class_labels.index(label)])
                X1_val.append(features[0])
                X2_val.append(spec[0])
                Y_val.append(config.class_labels.index(label))

        X1_test,X2_test,Y_test=[],[],[]
        test_feature = []
        for file in test_set: # only training samples need to be augmented
            file_name = os.path.basename(file)
            if config.dataset=="Crema":
                label = file_name.split('_')[2] 
                label = config.Emotions_map[file_name.split('_')[2]]
            elif config.dataset=="Ravdess":
                label = config.Emotions_map[file_name.split('-')[2]]
            elif config.dataset=="Savee":
                if file_name.split('_')[1][1].isnumeric():
                    label = config.Emotions_map[file_name.split('_')[1][0]]
                else:
                    label = config.Emotions_map[file_name.split('_')[1][:2]]
            print(label)
            if label in config.class_labels:
                features,spec = extract_features(config, file, max_,trainset=False) # features is a list of three lists
                test_feature.append([features[0],spec[0],config.class_labels.index(label)])
                X1_test.append(features[0])
                X2_test.append(spec[0])
                Y_test.append(config.class_labels.index(label))
        
        Y_train = np.array(Y_train)
        Y_val = np.array(Y_val)
        Y_test = np.array(Y_test)


        # if config.feature_folder does not exist，create a new one
        mkdirs(config.feature_folder)
        # path to save features
        feature_path_train = os.path.join(config.feature_folder, "train.p" if train == True else "predict.p")
        feature_path_val = os.path.join(config.feature_folder, "val.p" if train == True else "predict.p")
        feature_path_test = os.path.join(config.feature_folder, "test.p" if train == True else "predict.p")

        # save features
        pickle.dump(train_feature, open(feature_path_train, 'wb'))
        pickle.dump(val_feature, open(feature_path_val, 'wb'))
        pickle.dump(test_feature, open(feature_path_test, 'wb'))

    return X1_train,X2_train,Y_train,X1_val,X2_val,Y_val,X1_test,X2_test,Y_test




