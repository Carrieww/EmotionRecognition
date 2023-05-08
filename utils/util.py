import os
import torch
import random
import logging
import numpy as np
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
plt.ion()
# from model import ParallelModel

def k_fold_Split(data, n_splits, shuffle, random_state):
    # TODO: can change to RepeatedStratifiedKFold
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html#sklearn.model_selection.RepeatedStratifiedKFold
    # Creating k-fold splitting 
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # select all original data for data split
    x = data[data['indicator']=='Original'].reset_index(drop=True)
    y = x.loc[:,'emotion']
    for train_index, test_index in kfold.split(x,y):
        train = x.loc[train_index,:]
        test = x.loc[test_index,:]
        print("hello!")
    
        # training features should include AWGN augmented featured 
        train = data.merge(train.loc[:,"file_name"], on='file_name', how='inner')
        X1_train = np.array(train.loc[:,['audio_features']]['audio_features'].to_list())
        X2_train = np.array(train.loc[:,['spec']]['spec'].to_list())
        Z_train = np.array(train.loc[:,['video_features']]['video_features'].to_list())
        X1_test = np.array(test.loc[:,['audio_features']]['audio_features'].to_list())
        X2_test = np.array(test.loc[:,['spec']]['spec'].to_list())
        Z_test = np.array(test.loc[:,['video_features']]['video_features'].to_list())
        Y_train = np.array(train.loc[:,'emotion'].to_list())
        Y_test = np.array(test.loc[:,'emotion'].to_list())
        return X1_train,X2_train,Z_train,X1_test,X2_test,Z_test, Y_train, Y_test
    print()

def normalize_features(X_train,X_test,audio_features_flag=False):
    X_train = np.expand_dims(X_train,1)
    # X_val = np.expand_dims(X_val,1)
    X_test = np.expand_dims(X_test,1)

    scaler = StandardScaler()
    if audio_features_flag==False:
        b,c,h,w = X_train.shape
        X_train = np.reshape(X_train, newshape=(b,-1))
        X_train = scaler.fit_transform(X_train)
        print(X_train.shape)
        dump(scaler, '/home/wk1user1/Documents/metaAudio/Output/Ravdess_VO.joblib')
        X_train = np.reshape(X_train, newshape=(b,c,h,w))

        b,c,h,w = X_test.shape
        X_test = np.reshape(X_test, newshape=(b,-1))
        X_test = scaler.transform(X_test)
        X_test = np.reshape(X_test, newshape=(b,c,h,w))

    else:
        b,c,h = X_train.shape
        X_train = np.reshape(X_train, newshape=(b,-1))
        X_train = scaler.fit_transform(X_train)
        X_train = np.reshape(X_train, newshape=(b,c,h))

        b,c,h = X_test.shape
        X_test = np.reshape(X_test, newshape=(b,-1))
        X_test = scaler.transform(X_test)
        X_test = np.reshape(X_test, newshape=(b,c,h))
        
    return X_train,X_test

def mkdirs(folder_path: str) -> None:
    """Check the existence of folder, create one if not exist"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def setup_seed(config):
    random.seed(int(config.seed))
    np.random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))
    
def clean():
    torch.cuda.empty_cache()
    print("finished clean!")

def loss_plot(config, epoch,losses, test_losses) -> None:
    """plot losses versus test_losses"""
    fig = plt.figure(figsize=(12,8))
    x0=[i for i in range(epoch)]
    plt.title('Epochs & Train Test Losses',fontsize=18)
    plt.plot(x0,losses,'.-',label='Train Loss')
    plt.plot(x0,test_losses,'-',label='Test loss')
    plt.legend(prop={'size':16})
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Losses',fontsize=16)
    # plt.switch_backend('agg')
    plt.show()
    plt.savefig(os.path.join(config.dataset_path,config.loss_plot_path+'.png'))
    config.logger.info('Loss plot is saved to {}'.format(os.path.join(config.dataset_path,config.loss_plot_path+'.png')))
    # print('Loss plot is saved to {}'.format(os.path.join(config.dataset_path,config.loss_plot_path+'.png')))

def acc_plot(config, epoch,accuracies, test_accuracies) -> None:
    """plot training accuracy versus testing accuracy"""
    fig = plt.figure(figsize=(12,8))
    x0=[i for i in range(epoch)]
    plt.title('Epochs & Train Test Losses',fontsize=18)
    plt.plot(x0,accuracies,'.-',label='Train accuracies')
    plt.plot(x0,test_accuracies,'-',label='Test accuracies')
    plt.legend(prop={'size':16})
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    # plt.switch_backend('agg')
    plt.show()
    plt.savefig(os.path.join(config.dataset_path,config.acc_plot_path+'.png'))
    config.logger.info('Accuracy plot is saved to {}'.format(os.path.join(config.dataset_path,config.acc_plot_path+'.png')))
    # print('Accuracy plot is saved to {}'.format(os.path.join(config.dataset_path,config.acc_plot_path+'.png')))

def confusion_matrix_plot(config,Y_test,predictions):
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd

    predictions = predictions.numpy()
    cm = confusion_matrix(Y_test, predictions)
    names = []
    for ind in config.Emotions_map.keys():
        if config.Emotions_map[ind] in config.class_labels:
            names.append(config.Emotions_map[ind])
    df_cm = pd.DataFrame(cm, index=names, columns=names)

    if config.Emotion_Recognition_Mode == 'AV':
        Title = 'Confusion matrix of Audio-Visual Emotion Detection ('+config.dataset.upper()+')'
    elif config.Emotion_Recognition_Mode == 'AO':
        Title = 'Confusion matrix of SER ('+config.dataset.upper()+')'
    else:
        Title = 'Confusion matrix of FER ('+config.dataset.upper()+')'
    plt.figure(figsize=(12,8))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}) # font size
    plt.title(Title,fontdict={'fontsize': 20})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.savefig(os.path.join(config.dataset_path,config.confusion_matrix_path+'.png'))
    config.logger.info('Confusion Matrix plot is saved to {}'.format(os.path.join(config.dataset_path,config.confusion_matrix_path+'.png')))
    # print('Confusion Matrix plot is saved to {}'.format(config.confusion_matrix_path))

def logger(config):
    # Create and configure logger
    logging.basicConfig(filename=os.path.join(config.dataset_path,config.checkpoint_name+".log"),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    
    # Creating an object
    config.logger = logging.getLogger()
    
    # Setting the threshold of logger to DEBUG
    config.logger.setLevel(logging.INFO)


def saveModel(config,model,optimizer):
    os.makedirs('checkpoints',exist_ok=True)
    state={'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'seed':config.seed,
    }
    torch.save(state,os.path.join(config.dataset_path,config.checkpoint_name+'.pt'))
    config.logger.info('The checkpoint is saved to {}'.format(os.path.join(config.dataset_path,config.checkpoint_name+'.pt')))
    print('The checkpoint is saved to {}'.format(os.path.join(config.dataset_path,config.checkpoint_name+'.pt')))

def loadModel(config,model,optimizer):
    LOAD_PATH = os.path.join(os.getcwd(),config.dataset_path)
    # model = ParallelModel(len(config.Emotions_map),dropout=config.dropout,rnn_size=config.rnn_size).to(config.device)
    filename = os.path.join(LOAD_PATH,config.checkpoint_name+'.pt')
    if os.path.isfile(filename):
        config.logger.info("=> loading checkpoint '{}'".format(filename))
        # print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename,map_location=config.device)
        # epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        config.logger.info('Model is loaded from {}'.format(os.path.join(LOAD_PATH,config.checkpoint_name+'.pt')))
        # print('Model is loaded from {}'.format(os.path.join(LOAD_PATH,config.checkpoint_name+'.pt')))
        return model,optimizer
    else:
        config.logger.error("=> no checkpoint found at '{}'".format(filename))
        raise Exception("=> no checkpoint found at '{}'".format(filename))