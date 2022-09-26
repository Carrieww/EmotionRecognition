from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt
from model import ParallelModel
plt.ion()

def normalize_features(X_train,X_test,spec=True):
    X_train = np.expand_dims(X_train,1)
    # X_val = np.expand_dims(X_val,1)
    X_test = np.expand_dims(X_test,1)

    scaler = StandardScaler()
    if spec:
        b,c,h,w = X_train.shape
        X_train = np.reshape(X_train, newshape=(b,-1))
        X_train = scaler.fit_transform(X_train)
        X_train = np.reshape(X_train, newshape=(b,c,h,w))

        b,c,h,w = X_test.shape
        X_test = np.reshape(X_test, newshape=(b,-1))
        X_test = scaler.transform(X_test)
        X_test = np.reshape(X_test, newshape=(b,c,h,w))

        # b,c,h,w = X_val.shape
        # X_val = np.reshape(X_val, newshape=(b,-1))
        # X_val = scaler.transform(X_val)
        # X_val = np.reshape(X_val, newshape=(b,c,h,w))
    else:
        b,c,h = X_train.shape
        X_train = np.reshape(X_train, newshape=(b,-1))
        X_train = scaler.fit_transform(X_train)
        X_train = np.reshape(X_train, newshape=(b,c,h))

        b,c,h = X_test.shape
        X_test = np.reshape(X_test, newshape=(b,-1))
        X_test = scaler.transform(X_test)
        X_test = np.reshape(X_test, newshape=(b,c,h))

        # b,c,h = X_val.shape
        # X_val = np.reshape(X_val, newshape=(b,-1))
        # X_val = scaler.transform(X_val)
        # X_val = np.reshape(X_val, newshape=(b,c,h))
        
    return X_train,X_test#, X_val

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

def loss_plot(config, epoch,losses, val_losses) ->None:
    """plot losses versus val_losses"""
    fig = plt.figure(figsize=(12,8))
    x0=[i for i in range(epoch)]
    plt.title('Epochs & Train Val Losses',fontsize=18)
    plt.plot(x0,losses,'.-',label='Train Loss')
    plt.plot(x0,val_losses,'-',label='Val loss')
    plt.legend(prop={'size':16})
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Losses',fontsize=16)
    # plt.switch_backend('agg')
    plt.show()
    plt.savefig(config.loss_plot_path)
    print('Loss plot is saved to {}'.format(config.loss_plot_path))

def confusion_matrix_plot(config,Y_test,predictions):
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd

    predictions = predictions.cpu().numpy()
    cm = confusion_matrix(Y_test, predictions)
    names = []
    for ind in config.Emotions_map.keys():
        if config.Emotions_map[ind] in config.class_labels:
            names.append(config.Emotions_map[ind])
    df_cm = pd.DataFrame(cm, index=names, columns=names)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}) # font size
    plt.show()
    plt.savefig(config.confusion_matrix_path)
    print('Confusion Matrix plot is saved to {}'.format(config.confusion_matrix_path))

    
def saveModel(config,model,optimizer):
    os.makedirs('checkpoints',exist_ok=True)
    state={'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'seed':config.seed,
    }
    torch.save(state,os.path.join(config.checkpoint_path,config.checkpoint_name+'.pt'))
    print('Model is saved to {}'.format(os.path.join(config.checkpoint_path,config.checkpoint_name+'.pt')))

def loadModel(config,model,optimizer):
    LOAD_PATH = os.path.join(os.getcwd(),'checkpoints')
    # model = ParallelModel(len(config.Emotions_map),dropout=config.dropout,rnn_size=config.rnn_size).to(config.device)
    filename = os.path.join(LOAD_PATH,config.checkpoint_name+'.pt')
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename,map_location=config.device)
        # epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Model is loaded from {}'.format(os.path.join(LOAD_PATH,config.checkpoint_name+'.pt')))
        return model,optimizer
    else:
        raise Exception("=> no checkpoint found at '{}'".format(filename))