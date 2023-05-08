from utils.opts import parse_opt
from utils.util import normalize_features,loss_plot,clean,setup_seed,saveModel,loadModel,confusion_matrix_plot, logger,acc_plot,k_fold_Split
from feature_extraction import get_features,load_features
from model import AV_transformer,AO_transformer, VO_transformer, make_train_step,make_validate_fnc,loss_fnc
import numpy as np
import torch
import time
import os
from sklearn.model_selection import StratifiedKFold,train_test_split

def main(train=True):
    clean()
    config = parse_opt()
    setup_seed(config)
    logger(config)

    if config.dataset=="Crema":
        config.Emotions_map= {'NEU':'neutral',
        'HAP':'happy',
        'SAD':'sad',
        'ANG':'angry',
        'FEA':'fearful',
        'DIS':'disgust'}
    elif config.dataset=="Ravdess":
        config.Emotions_map = {
            '01':'neutral',
            '02':'calm',
            '03':'happy',
            '04':'sad',
            '05':'angry',
            '06':'fearful',
            '07':'disgust',
            '08':'surprised'
        }
    elif config.dataset=="Savee":
        config.Emotions_map = {
            'n':'neutral',
            'h':'happy',
            'sa':'sad',
            'a':'angry',
            'f':'fearful',
            'd':'disgust',
            'su':'surprised'
        }
    elif config.dataset=="RML":
        config.Emotions_map = {
            'ha':'happy',
            'sa':'sad',
            'an':'angry',
            'fe':'fearful',
            'di':'disgust',
            'su':'surprised'
        }

    print(torch.cuda.is_available())
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.logger.info('Selected device is {}'.format(config.device))

    # audio-only, video-only, and audio-visual 
    if config.Emotion_Recognition_Mode =='AV':
        model = AV_transformer(num_emotions=len(config.class_labels),dropout=config.dropout,dim_feedforward=config.dim_feedforward, num_layers=config.num_layers).to(config.device)
    elif config.Emotion_Recognition_Mode == 'AO':
        model = AO_transformer(num_emotions=len(config.class_labels),dropout=config.dropout,dim_feedforward=config.dim_feedforward, num_layers=config.num_layers).to(config.device)
    elif config.Emotion_Recognition_Mode =='VO':
        model = VO_transformer(num_emotions=len(config.class_labels),dropout=config.dropout,dim_feedforward=config.dim_feedforward, num_layers=config.num_layers).to(config.device)
    OPTIMIZER = torch.optim.AdamW(model.parameters(),lr=config.lr, weight_decay=config.weight_decay)

    # training or testing 
    if config.mode == "train":

        config.t_initial = time.time()
        # model initiation
        if config.pretrained == True:
            model,OPTIMIZER = loadModel(config,model,OPTIMIZER)
            config.logger.info('Finished preloading! Pretrained model is used.')
            print('Finished preloading! Pretrained model is used.')
        train_step = make_train_step(model, loss_fnc, optimizer=OPTIMIZER)
        validate = make_validate_fnc(model,loss_fnc, config)

        # get raw features
        if  (os.path.isfile(os.path.join(config.feature_folder,config.dataset, "dataset.p"))== False):
            get_features(config)            
        df = load_features(config,"dataset.p")
        
        # k-fold cross validation
        kfold = StratifiedKFold(n_splits=config.K_Fold, shuffle=True, random_state=config.seed)
        x_train = df[df['indicator']=='Original'].reset_index(drop=True)
        y_train = x_train.loc[:,'emotion']

        if config.cross_validation == False:
            stop_index = 0 # stop at the first spliting, no cv
        else:
            stop_index = config.K_Fold-1
        best_accuracy = 0.0
        config.losses,config.accuracies=[],[]
        config.test_losses,config.test_accuracies = [],[]
        config.best_acc_cv = []
        for i, (train_index, test_index) in enumerate(kfold.split(x_train,y_train)):
            t0=time.time()
            train = x_train.loc[train_index,:]
            test = x_train.loc[test_index,:]

            # get audio (X1 - 22 manually extracted audio features, X2 - mel spectragram) features and facial (Z) features
            train = df.merge(train.loc[:,"file_name"], on='file_name', how='inner')
            X1_train = np.array(train.loc[:,['audio_features']]['audio_features'].to_list())
            X2_train = np.array(train.loc[:,['spec']]['spec'].to_list())
            Z_train = np.array(train.loc[:,['video_features']]['video_features'].to_list())
            X1_test = np.array(test.loc[:,['audio_features']]['audio_features'].to_list())
            X2_test = np.array(test.loc[:,['spec']]['spec'].to_list())
            Z_test = np.array(test.loc[:,['video_features']]['video_features'].to_list())
            Y_train = np.array(train.loc[:,'emotion'].to_list())
            Y_test = np.array(test.loc[:,'emotion'].to_list())

            # feature normalization
            X1_train,X1_test = normalize_features(X1_train,X1_test,audio_features_flag=True)
            X2_train,X2_test = normalize_features(X2_train,X2_test)
            Z_train,Z_test = normalize_features(Z_train,Z_test)
            config.DATASET_SIZE = X1_train.shape[0]

            model, OPTIMIZER, best_accuracy = start_training(i,config,model, OPTIMIZER, train_step, validate, X1_train,X2_train,Z_train,X1_test,X2_test,Z_test, Y_train, Y_test,t0, best_accuracy)
            config.logger.info(f"Time per CV: {(time.time()-t0):.4f}")

            if i == stop_index:
                # plot the loss and accuracy changes of validation data in cross validation
                loss_plot(config, len(config.losses),config.losses, config.test_losses)
                acc_plot(config, len(config.losses),config.accuracies, config.test_accuracies)
                break

        # finish all cv training
        config.logger.info(f'Finished training with CV! Total Time: {(time.time()-config.t_initial):.4f}, average accuracy across CV is: {np.mean(config.best_acc_cv)}%')
        print(f'Finished training with CV! Total Time: {(time.time()-config.t_initial):.4f}, average accuracy across CV is: {np.mean(config.best_acc_cv)}%')

    elif config.mode == "test":
        pass

def start_training(CV_split_index,config,model, OPTIMIZER, train_step, validate, X1_train,X2_train,Z_train,X1_test,X2_test,Z_test, Y_train, Y_test,t0, best_accuracy):
    for epoch in range(config.epochs):
        best_accuracy_per_cv = 0.0
        t_epoch=time.time()
        # schuffle data
        ind = np.random.permutation(config.DATASET_SIZE)
        X1_train = X1_train[ind,:,:]
        X2_train = X2_train[ind,:,:,:]
        Z_train = Z_train[ind,:,:,:]
        Y_train = Y_train[ind]
        epoch_acc = 0.0
        epoch_loss = 0.0
        iters = int(config.DATASET_SIZE / config.batch_size)
        for i in range(iters):
            batch_start = i * config.batch_size
            batch_end = min(batch_start + config.batch_size, config.DATASET_SIZE)
            actual_batch_size = batch_end-batch_start
            X1 = X1_train[batch_start:batch_end,:,:]
            X2 = X2_train[batch_start:batch_end,:,:,:]
            Z = Z_train[batch_start:batch_end,:,:,:]
            Y = Y_train[batch_start:batch_end]
            X1_tensor = torch.tensor(X1,device=config.device).float()
            X2_tensor = torch.tensor(X2,device=config.device).float()
            Z_tensor = torch.tensor(Z,device=config.device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long,device=config.device)
            loss, acc = train_step(X1_tensor,X2_tensor,Z_tensor, Y_tensor)
            epoch_acc += acc*actual_batch_size/config.DATASET_SIZE
            epoch_loss += loss*actual_batch_size/config.DATASET_SIZE
            config.logger.info(f"\r Epoch {epoch+1}: iteration {i+1}/{iters}")
            print(f"\r Epoch {epoch+1}: iteration {i+1}/{iters}",end='\n')

        X1_test_tensor = torch.tensor(X1_test,device=config.device).float()
        X2_test_tensor = torch.tensor(X2_test,device=config.device).float()
        Z_test_tensor = torch.tensor(Z_test,device=config.device).float()
        Y_test_tensor = torch.tensor(Y_test,dtype=torch.long,device=config.device)
        test_loss, test_acc, test_predictions = validate(X1_test_tensor,X2_test_tensor, Z_test_tensor,Y_test_tensor)

        config.losses.append(epoch_loss)
        config.test_losses.append(test_loss)
        config.accuracies.append(acc.cpu())
        config.test_accuracies.append(test_acc.cpu())
        config.logger.info('')
        config.logger.info(f"CV Split {CV_split_index+1} --> Epoch {epoch+1} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, test_loss:{test_loss:.4f}, test_acc:{test_acc:.2f}%, Time per epoch: {(time.time()-t_epoch):.4f}")
        print('')
        print(f"CV Split {CV_split_index+1} --> Epoch {epoch+1} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, test_loss:{test_loss:.4f}, test_acc:{test_acc:.2f}%, Time per epoch: {(time.time()-t_epoch):.4f}")

        # save the model with the highest test accuracy
        if test_acc>best_accuracy:
            best_accuracy = test_acc
            saveModel(config,model,OPTIMIZER)
            config.logger.info(f"The best model up to now is saved with test loss:{test_loss}% and test accuracy:{test_acc:.2f}% at epoch: {epoch+1}/{config.epochs}")
            print(f"The best model up to now is saved with test loss:{test_loss}% and test accuracy:{test_acc:.2f}% at epoch: {epoch+1}/{config.epochs}")
        if test_acc > best_accuracy_per_cv:
            best_accuracy_per_cv = test_acc
            

        # Print time spend in the end of each training with cross validation
        if epoch == config.epochs - 1:
            config.best_acc_cv.append(best_accuracy_per_cv.cpu())
            config.logger.info(f'Finished training per data split! Total Time: {(time.time()-t0):.4f}; best acc per cv is: {best_accuracy_per_cv}%')
            confusion_matrix_plot(config,Y_test_tensor.cpu(),test_predictions.cpu())
            print(f'Finished training per data split! Total Time: {(time.time()-t0):.4f}; best acc per cv is: {best_accuracy_per_cv}%')

    return model, OPTIMIZER, best_accuracy

if __name__ == '__main__':
    train=True
    main(train)