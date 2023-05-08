
import torch
import torch.nn as nn
import torchmetrics

class AV_transformer(nn.Module):
    def __init__(self,num_emotions,dropout,dim_feedforward,num_layers):
        super().__init__()
        # Audio Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2,4], stride=[2,4])
        transf_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=dim_feedforward, dropout=dropout, activation='relu')
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=num_layers)

        # Facial Transformer block
        self.transf_maxpool_facial = nn.MaxPool2d(kernel_size=[1,4], stride=[1,2])
        transf_layer_facial = nn.TransformerEncoderLayer(d_model=35, nhead=5, dim_feedforward=dim_feedforward, dropout=dropout, activation='relu')
        self.transf_encoder_facial = nn.TransformerEncoder(transf_layer_facial, num_layers=num_layers)

        # Linear softmax layer
        # self.linear_in = hidden_size
        self.out_linear = nn.LazyLinear(num_emotions) #2*hidden_size+192,
        self.dropout_linear = nn.Dropout(p=dropout)
        self.out_softmax = nn.Softmax(dim=1)
    def forward(self,x1,x2,z):

        # transformer embedding for audios
        x_reduced = self.transf_maxpool(x2)
        x_reduced = torch.squeeze(x_reduced,1)
        x_reduced = x_reduced.permute(2,0,1) # requires shape = (time,batch,embedding)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)

        # transformer embedding for videos
        z_reduced = self.transf_maxpool_facial(z)
        z_reduced = torch.squeeze(z_reduced,1)
        z_reduced = z_reduced.permute(2,0,1) # requires shape = (time,batch,embedding)
        z_transf_out = self.transf_encoder_facial(z_reduced)
        z_transf_embedding = torch.mean(z_transf_out, dim=0)


        # concatenate with extracted features
        complete_embedding = torch.cat([transf_embedding,z_transf_embedding,x1.squeeze()],dim=1)
        
        # self.linear_in = complete_embedding.shape[1]
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax#, attention_weights_norm

class VO_transformer(nn.Module):
    def __init__(self,num_emotions,dropout,dim_feedforward,num_layers):
        super().__init__()
        # Facial Transformer block
        self.transf_maxpool_facial = nn.MaxPool2d(kernel_size=[1,4], stride=[1,2])
        transf_layer_facial = nn.TransformerEncoderLayer(d_model=35, nhead=5, dim_feedforward=dim_feedforward, dropout=dropout, activation='relu')
        self.transf_encoder_facial = nn.TransformerEncoder(transf_layer_facial, num_layers=num_layers)

        # Linear softmax layer
        # self.linear_in = hidden_size
        self.out_linear = nn.LazyLinear(num_emotions) #2*hidden_size+192,
        self.dropout_linear = nn.Dropout(p=dropout)
        self.out_softmax = nn.Softmax(dim=1)
    def forward(self,x1,x2,z):

        # transformer embedding for videos
        z_reduced = self.transf_maxpool_facial(z)
        z_reduced = torch.squeeze(z_reduced,1)
        z_reduced = z_reduced.permute(2,0,1) # requires shape = (time,batch,embedding)
        z_transf_out = self.transf_encoder_facial(z_reduced)
        z_transf_embedding = torch.mean(z_transf_out, dim=0)


        # concatenate with extracted features
        complete_embedding = z_transf_embedding
        
        # self.linear_in = complete_embedding.shape[1]
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax#, attention_weights_norm

class AO_transformer(nn.Module):
    def __init__(self,num_emotions,dropout, dim_feedforward,num_layers):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(in_channels=1,
                       out_channels=16,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),
            # 2. conv block
            nn.Conv2d(in_channels=16,
                       out_channels=32,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),
            # 3. conv block
            # nn.Conv2d(in_channels=32,
            #            out_channels=64,
            #            kernel_size=3,
            #            stride=1,
            #            padding=1
            #           ),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=dropout),
            # # 4. conv block
            # nn.Conv2d(in_channels=64,
            #            out_channels=64,
            #            kernel_size=3,
            #            stride=1,
            #            padding=1
            #           ),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=dropout),
            # 5. conv block
            # nn.Conv2d(in_channels=64,
            #            out_channels=128,
            #            kernel_size=3,
            #            stride=1,
            #            padding=1
            #           ),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=dropout)
        )

        # Audio Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2,4], stride=[2,4])
        transf_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=dim_feedforward, dropout=dropout, activation='relu')
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=num_layers)

        # Linear softmax layer
        # self.linear_in = hidden_size
        self.out_linear = nn.LazyLinear(num_emotions) #2*hidden_size+192,
        self.dropout_linear = nn.Dropout(p=dropout)
        self.out_softmax = nn.Softmax(dim=1)
    def forward(self,x1,x2,z):
        # audio features
        # conv embedding
        # conv_embedding = self.conv2Dblock(x2) # (b,channel,freq,time)
        # conv_embedding = torch.flatten(conv_embedding, start_dim=1) # do not flatten batch dimension

        # transformer embedding
        x_reduced = self.transf_maxpool(x2)
        x_reduced = torch.squeeze(x_reduced,1)
        x_reduced = x_reduced.permute(2,0,1) # requires shape = (time,batch,embedding)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)

        # concatenate
        complete_embedding = torch.cat([transf_embedding,x1.squeeze()],dim=1)
        
        # self.linear_in = complete_embedding.shape[1]
        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax#, attention_weights_norm


def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)

def make_train_step(model, loss_fnc, optimizer):
    def train_step(X1,X2,Z,Y):
        # set model to train mode
        model.train()
        # forward pass
        output_logits, output_softmax = model(X1,X2,Z)
        predictions = torch.argmax(output_softmax,dim=1)
        accuracy = torch.sum(Y==predictions)/float(len(Y))
        # compute loss
        loss = loss_fnc(output_logits, Y)
        # compute gradients
        
        loss.backward()
        # update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy*100
    return train_step

def make_validate_fnc(model,loss_fnc,config):
    def validate(X1,X2,Z,Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X1,X2,Z)
            predictions = torch.argmax(output_softmax,dim=1)
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            confmat_ = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=len(config.class_labels)).to(config.device)
            confmat=confmat_(preds=predictions, target=Y.int())
            config.logger.info(confmat)
            print(confmat)


            # accuracy_new = (confmat[0][0] + confmat[1][1]+confmat[2][2]+confmat[3][3]+confmat[4][4]+confmat[5][5]) / torch.sum(confmat)
            # recall = confmat[1][1] / (confmat[1][1] + confmat[1][0])
            # precision = confmat[1][1] / (confmat[1][1] + confmat[0][1])
            # print(accuracy==accuracy_new)
            # print(recall)
            # print(precision)

            # f1_score = torchmetrics.F1Score(task="multiclass", num_classes=6, average=None).to(config.device)
            # F1 = f1_score(preds=predictions, target=Y)[1]
            # print(F1)

            loss = loss_fnc(output_logits,Y)
        return loss.item(), accuracy*100, predictions
    return validate