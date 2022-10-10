from utils.opts import parse_opt
from utils.util import normalize_features,loss_plot,clean,setup_seed,saveModel,loadModel,confusion_matrix_plot, logger,acc_plot
from feature_extraction import get_features,load_features
from model import AV_transformer,AO_transformer, VO_transformer, make_train_step,make_validate_fnc,loss_fnc
import numpy as np
import torch
import time

def main():
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

    t0=time.time()
    if config.feature_exist == True:
        X1_train,X2_train,Z_train, Y_train, G_train = load_features(config,"train.p")
        X1_test,X2_test,Z_test, Y_test, G_test = load_features(config,"test.p")
    else:
        get_features(config, train=True)
        X1_train,X2_train,Z_train, Y_train, G_train = load_features(config,"train.p")
        X1_test,X2_test,Z_test, Y_test, G_test = load_features(config,"test.p")

    X1_train,X1_test = normalize_features(X1_train,X1_test,audio_features_flag=True)
    X2_train,X2_test = normalize_features(X2_train,X2_test)
    Z_train,Z_test = normalize_features(Z_train,Z_test)

    DATASET_SIZE = X1_train.shape[0]
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.logger.info('Selected device is {}'.format(config.device))

    if config.mode =='AV':
        model = AV_transformer(num_emotions=len(config.Emotions_map),dropout=config.dropout,dim_feedforward=config.dim_feedforward).to(config.device)
    elif config.mode == 'AO':
        model = AO_transformer(num_emotions=len(config.Emotions_map),dropout=config.dropout,dim_feedforward=config.dim_feedforward).to(config.device)
    elif config.mode =='VO':
        model = VO_transformer(num_emotions=len(config.Emotions_map),dropout=config.dropout,dim_feedforward=config.dim_feedforward).to(config.device)
    OPTIMIZER = torch.optim.AdamW(model.parameters(),lr=config.lr, weight_decay=config.weight_decay)

    if config.pretrained == True:
        model,OPTIMIZER = loadModel(config,model,OPTIMIZER)
        config.logger.info('Finished preloading! Pretrained model is used.')
        print('Finished preloading! Pretrained model is used.')

    train_step = make_train_step(model, loss_fnc, optimizer=OPTIMIZER)
    validate = make_validate_fnc(model,loss_fnc)

    # start training
    best_accuracy = 0.0
    losses,accuracies=[],[]
    test_losses,test_accuracies = [],[]
    for epoch in range(config.epochs):
        t_epoch=time.time()
        # schuffle data
        ind = np.random.permutation(DATASET_SIZE)
        X1_train = X1_train[ind,:,:]
        X2_train = X2_train[ind,:,:,:]
        Z_train = Z_train[ind,:,:,:]
        Y_train = Y_train[ind]
        epoch_acc = 0.0
        epoch_loss = 0.0
        iters = int(DATASET_SIZE / config.batch_size)
        for i in range(iters):
            batch_start = i * config.batch_size
            batch_end = min(batch_start + config.batch_size, DATASET_SIZE)
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
            epoch_acc += acc*actual_batch_size/DATASET_SIZE
            epoch_loss += loss*actual_batch_size/DATASET_SIZE
            config.logger.info(f"\r Epoch {epoch+1}: iteration {i+1}/{iters}")
            print(f"\r Epoch {epoch+1}: iteration {i+1}/{iters}",end='')

        X1_test_tensor = torch.tensor(X1_test,device=config.device).float()
        X2_test_tensor = torch.tensor(X2_test,device=config.device).float()
        Z_test_tensor = torch.tensor(Z_test,device=config.device).float()
        Y_test_tensor = torch.tensor(Y_test,dtype=torch.long,device=config.device)
        test_loss, test_acc, test_predictions = validate(X1_test_tensor,X2_test_tensor, Z_test_tensor,Y_test_tensor)

        losses.append(epoch_loss)
        test_losses.append(test_loss)
        accuracies.append(acc)
        test_accuracies.append(test_acc)
        config.logger.info('')
        config.logger.info(f"Epoch {epoch+1} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, test_loss:{test_loss:.4f}, test_acc:{test_acc:.2f}%, Time per epoch: {(time.time()-t_epoch):.4f}")
        print('')
        print(f"Epoch {epoch+1} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, test_loss:{test_loss:.4f}, test_acc:{test_acc:.2f}%, Time per epoch: {(time.time()-t_epoch):.4f}")

        # save the model with the highest test accuracy
        if test_acc>best_accuracy:
            best_accuracy = test_acc
            saveModel(config,model,OPTIMIZER)
            config.logger.info(f"The best model up to now is saved with test accuracy:{test_acc:.2f}% at epoch: {epoch+1}/{config.epochs}")
            print(f"The best model up to now is saved with test accuracy:{test_acc:.2f}% at epoch: {epoch+1}/{config.epochs}")

        # save all plots, confusion matrix and overall time spend in the end
        if epoch == config.epochs - 1:
            confusion_matrix_plot(config,Y_test,test_predictions)
            loss_plot(config, epoch+1,losses, test_losses)
            acc_plot(config, epoch+1,accuracies, test_accuracies)
            config.logger.info(f'Finished training! Total Time: {(time.time()-t0):.4f}')
            print(f'Finished training! Total Time: {(time.time()-t0):.4f}')



if __name__ == '__main__':
    main()