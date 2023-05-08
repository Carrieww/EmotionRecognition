import os
import math
import pandas as pd
from zipfile import ZipFile
from sklearn.model_selection import StratifiedShuffleSplit
from random import shuffle
import numpy as np
import librosa
import glob
import pickle
import joblib

from utils.util import mkdirs

def load_features(config, file_name: str):
    """
    Load features from "{config.feature_folder}/*.p"
    Args:
        config:
        train (bool):
    Returns:
        - X1 ([np.ndarray]): audio features
        - X2 ([np.ndarray]): audio spec features
        - Z ([np.ndarray]): video features
        - Y ([np.ndarray]): labels
    """
    feature_path = os.path.join(config.feature_folder,config.dataset, file_name)

    df = pd.DataFrame(
        data = joblib.load(feature_path),
        columns = ['audio_features','spec', 'indicator','video_features','emotion','file_name']
    )

    # df['indicator'] = pd.factorize(df['indicator'])[0]
    # df['file_name'] = pd.factorize(df['file_name'])[0]

    # X1 = list(features['audio_features'])
    # X1_Category = list(features['indicator'])
    # X2 = list(features['spec'])
    # Z = list(features['video_features'])
    # Y = list(features['emotion'])
    # Y = np.array(Y)
    # G = list(features['file_name'])
    # # convert File_names and X1_Category into array of int
    # G_data = {"File_name": G}
    # G = pd.factorize(G_data['File_name'])[0]
    # X1_Category_data = {"X1_Category":X1_Category}
    # X1_Category = pd.factorize(X1_Category_data['X1_Category'])[0]

    return df


def get_data_path(config):

    with ZipFile(config.Zip_file_path, 'r') as zipfile:
        Name_list = np.array(zipfile.namelist())
        Audio_files = Name_list[[x.endswith(".wav") and x.startswith(config.dataset) for x in Name_list]]
        Video_files = Name_list[[x.endswith(".csv") and x.startswith(config.dataset) for x in Name_list]]
    
    config.logger.info('The dataset has '+str(len(Audio_files))+' audio samples and '+str(len(Video_files))+' video files.')
    assert len(Audio_files)==len(Video_files), f"number of audio files and video files are expected to be the same, got different."
    
    if config.dataset=='Ravdess':
        assert len(Audio_files) == 1440, f"number of samples mismatches 1440 for {config.dataset}, got: {len(Audio_files)}"
    elif config.dataset =='Savee':
        assert len(Audio_files)==480, f"number of samples mismatches 480 for {config.dataset}, got: {len(Audio_files)}"
    elif config.dataset =='RML':
        assert len(Audio_files)==720, f"number of samples mismatches 720 for {config.dataset}, got: {len(Audio_files)}"
    # if config.data_size != 'None':
    #     Audio_files=Audio_files[:config.data_size]
    #     Video_files=Video_files[:config.data_size]

    temp = list(zip(Audio_files, Video_files))
    shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    res1, res2 = list(res1), list(res2)

    res1=np.array(res1)
    res2=np.array(res2)
    files=np.vstack((res1,res2))

    return files

def get_audio_min_max(config,files):

    if len(files[0].tolist())!=0:
        files=files[0].tolist() # extract audio files
    else:
        config.logger.error('No audio wav files detected, so there is no min max of audio.')

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
    config.logger.info('The min length of audio file is '+str(min_))
    config.logger.info('The max length of audio file is '+str(max_))
    config.logger.info('The audio file with the maximum length comes from '+str(max_sf))

    return min_,max_

def get_video_min_max(config,files):

    if len(files[1].tolist())!=0:
        files=files[1].tolist() # extract video csv files
    else:
        config.logger.error('No video csv files detected, so there is no min max of video.')

    min_,max_ = 10000,0
    for file in files:
        with ZipFile(config.Zip_file_path, 'r').open(file) as f:
            df=pd.read_csv(f)
            if df.shape[0] < min_:
                min_ = df.shape[0]
            elif df.shape[0] > max_:
                max_ = df.shape[0]
                max_sf=file
    config.logger.info('The min length of video file is '+str(min_))
    config.logger.info('The max length of video file is '+str(max_))
    config.logger.info('The video file with the maximum length comes from '+str(max_sf))

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

def extract_audio_features(config,file,max_,trainset):
    file=file+'.wav'
    with ZipFile(config.Zip_file_path, 'r').open(file) as f:
        X,sr=librosa.load(f,sr=config.sample_rate)
    max_x = X.shape[0] / sr
    if max_x < max_:
        length = (max_ * sr) - X.shape[0]
        left_pad = math.floor(length/2)
        right_pad = math.ceil(length/2)
        X = np.pad(X, ((left_pad,right_pad)), 'constant')

    if trainset == True:
        augmented_signals = addAWGN(X)
        ext_features=[]
        ext_spec = []
        indicator = []
        for i in range(augmented_signals.shape[0]):
            ext_features_aug,ext_spec_aug = features(augmented_signals[i], sr,1024,256)
            ext_features.append(ext_features_aug)
            ext_spec.append(ext_spec_aug)
            indicator.append("AWGN")
        ext_features_ori,ext_spec_ori = features(X, sr,1024,256)
        ext_features.append(ext_features_ori)
        ext_spec.append(ext_spec_ori)
        indicator.append("Original")
    else:
        ext_features,ext_spec = features(X, sr,1024,256)
        ext_features = [ext_features]
        ext_spec = [ext_spec]
    return ext_features,ext_spec,indicator

def extract_video_features(config,file,max_):
    file=file+'.csv'
    with ZipFile(config.Zip_file_path, 'r').open(file) as f:
        df=pd.read_csv(f)
    df.columns = df.columns.str.replace(' ', '')
    df=df[config.video_features]

    assert df.shape[1]==len(config.video_features), "extracted fewer/more video features"
    max_x = df.shape[0]

    if max_x < max_:
        length = max_-max_x
        X = np.pad(df.values, ((0,int(length)), (0,0)), 'constant')
    else:
        X = df.values
    # X=X.T
    # print(X.shape)
    return X.T # shape = video_features * t

def get_filenames(config,files):
    # extract only audio files because we only need file names
    if files.shape[0]==2:
        files=files[0].tolist() 
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
        elif config.dataset=="RML":
            label = file_name.split('_')[len(file_name.split('_'))-1].split('.')[0]
            label = config.Emotions_map[label[:2]]
        if label in config.class_labels:
            Y.append(label)
            new_files.append(os.path.join(os.path.dirname(file),file_name.split('.')[0]))
            
    new_files=np.array(new_files)
    # sss = StratifiedShuffleSplit(test_size=config.test_ratio, random_state=config.seed)
    # for train_index, test_index in sss.split(new_files,Y):
    #     train_set, test_set = new_files[train_index], new_files[test_index]
    
    return new_files

def get_features(config):

    files = get_data_path(config) # files are with .wav and .csv

    _,max_audio = get_audio_min_max(config,files)
    _,max_video = get_video_min_max(config,files)

    # files are on longer with .wav and .csv
    data_set = get_filenames(config,files) 


    # get X_train and Y_train, with augmentation
    dataset_feature = []
    for file in data_set: # only training samples need to be augmented
        if config.dataset=="Crema":
            label = file.split('_')[2]                 
            label = config.Emotions_map[file.split('_')[2]]
            file_name = file #.split('_')[0]
        elif config.dataset=="Ravdess":
            label = config.Emotions_map[file.split('-')[2]]
            file_name = file#.split('-')[6].split('.')[0]
        elif config.dataset=="Savee":
            file_name = file#file.split('_')[0]
            if file.split('_')[len(file.split('_'))-1][1].isnumeric():
                label = config.Emotions_map[file.split('_')[len(file.split('_'))-1][0]]
            else:
                label = config.Emotions_map[file.split('_')[len(file.split('_'))-1][:2]]
        elif config.dataset=="RML":
            label = file.split('_')[len(file.split('_'))-1].split('.')[0]
            label = config.Emotions_map[label[:2]]
            file_name = file#.split('_')[0]
        if label in config.class_labels:
            # print('train')
            print(label)
            audio_features,audio_spec,indicator = extract_audio_features(config, file, max_audio,True) # features is a list of three lists
            video_features = extract_video_features(config, file, max_video)
            for i in range(len(audio_features)):
                dataset_feature.append([audio_features[i],audio_spec[i],indicator[i],video_features, config.class_labels.index(label),file_name])

    # test_feature = []
    # for file in test_set: # only training samples need to be augmented
    #     if config.dataset=="Crema":
    #         label = file.split('_')[2] 
    #         label = config.Emotions_map[file.split('_')[2]]
    #         file_name = file.split('_')[0]
    #     elif config.dataset=="Ravdess":
    #         label = config.Emotions_map[file.split('-')[2]]
    #         file_name = file.split('-')[6].split('.')[0]
    #     elif config.dataset=="Savee":
    #         file_name = file.split('_')[0]
    #         if file.split('_')[1][1].isnumeric():
    #             label = config.Emotions_map[file.split('_')[1][0]]
    #         else:
    #             label = config.Emotions_map[file.split('_')[1][:2]]
    #     elif config.dataset=="RML":
    #         label = file.split('_')[len(file.split('_'))-1].split('.')[0]
    #         label = config.Emotions_map[label[:2]]
    #         file_name = file.split('_')[0]
    #     if label in config.class_labels:
    #         # print('train')
    #         print(label)
    #         audio_features,audio_spec,indicator = extract_audio_features(config, file, max_audio,False) # features is a list of three lists
    #         video_features = extract_video_features(config, file, max_video)
    #         test_feature.append([audio_features[0],audio_spec[0],video_features, config.class_labels.index(label),group])

    # path to save features
    feature_path_dataset = os.path.join(config.feature_folder,config.dataset, "dataset.p")

    # save features
    pickle.dump(dataset_feature, open(feature_path_dataset, 'wb'))

    config.logger.info("Finished extracting features!")
    print("Finished extracting features!")


