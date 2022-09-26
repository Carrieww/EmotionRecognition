from utils.opts import parse_opt
from utils.util import normalize_features,loss_plot,clean,setup_seed,saveModel,loadModel,confusion_matrix_plot
from feature_extraction import get_features,load_features
from model import ParallelModel,make_train_step,make_validate_fnc,loss_fnc
import numpy as np
import torch
import time

def main():
    clean()
    config = parse_opt()
    setup_seed(config)

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
        X1_train,X2_train,Y_train = load_features(config,"train.p")
        # X1_val,X2_val,Y_val = load_features(config,"val.p")
        X1_test,X2_test,Y_test = load_features(config,"test.p")
    else:
        X1_train,X2_train,Y_train,X1_val,X2_val,Y_val,X1_test,X2_test,Y_test = get_features(config, train=True)

    X1_train,X1_test = normalize_features(X1_train,X1_test,spec=False)
    X2_train,X2_test = normalize_features(X2_train,X2_test)

    DATASET_SIZE = X1_train.shape[0]
    config.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
    print('Selected device is {}'.format(config.device))
    model = ParallelModel(num_emotions=len(config.Emotions_map),dropout=config.dropout,rnn_size=config.rnn_size).to(config.device)
    OPTIMIZER = torch.optim.AdamW(model.parameters(),lr=config.lr, weight_decay=config.weight_decay)

    if config.pretrained == True:
        model,OPTIMIZER = loadModel(config,model,OPTIMIZER)
        print('Finished preloading!')

    train_step = make_train_step(model, loss_fnc, optimizer=OPTIMIZER)
    validate = make_validate_fnc(model,loss_fnc)
    losses=[]
    val_losses = []
    for epoch in range(config.epochs):
        t_epoch=time.time()
        # schuffle data
        ind = np.random.permutation(DATASET_SIZE)
        X1_train = X1_train[ind,:,:]
        X2_train = X2_train[ind,:,:,:]
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
            Y = Y_train[batch_start:batch_end]
            X1_tensor = torch.tensor(X1,device=config.device).float()
            X2_tensor = torch.tensor(X2,device=config.device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long,device=config.device)
            loss, acc = train_step(X1_tensor,X2_tensor,Y_tensor)
            epoch_acc += acc*actual_batch_size/DATASET_SIZE
            epoch_loss += loss*actual_batch_size/DATASET_SIZE
            print(f"\r Epoch {epoch}: iteration {i}/{iters}",end='')

        # X1_val_tensor = torch.tensor(X1_val,device=config.device).float()
        # X2_val_tensor = torch.tensor(X2_val,device=config.device).float()
        # Y_val_tensor = torch.tensor(Y_val,dtype=torch.long,device=config.device)
        X1_test_tensor = torch.tensor(X1_test,device=config.device).float()
        X2_test_tensor = torch.tensor(X2_test,device=config.device).float()
        Y_test_tensor = torch.tensor(Y_test,dtype=torch.long,device=config.device)
        val_loss, val_acc, _ = 0,0,0 #validate(X1_val_tensor,X2_val_tensor,Y_val_tensor)
        test_loss, test_acc, test_predictions = validate(X1_test_tensor,X2_test_tensor,Y_test_tensor)

        losses.append(epoch_loss)
        val_losses.append(test_loss)
        print('')
        print(f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%, Time per epoch: {(time.time()-t_epoch):.4f}")
        print(f"--- --- --> test_loss:{test_loss:.4f}, test_acc:{test_acc:.2f}%")
        if epoch == config.epochs - 1:
            print(len(Y_test))
            print(len(test_predictions))
            confusion_matrix_plot(config,Y_test,test_predictions)
            loss_plot(config, epoch+1,losses, val_losses)
            saveModel(config,model,OPTIMIZER)
            print(f'Model saved! Total Time: {(time.time()-t0):.4f}')



if __name__ == '__main__':
    main()