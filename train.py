import torch
from torch.utils.data import DataLoader
import numpy as np
from config import Config
from data_loader import TrainDataset
from models import Autoencoder
from utils import *
from loss import SSIM

def train_one_step(model,criterion, optimizer,data):
    image_batch = data["image"].cuda()
    image_batch = image_batch.float().cuda()

    # ===================forward=====================
    output = model(image_batch)
    loss = criterion(output, image_batch)
    
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def validate_one_step(model,criterion,data):
    image_batch = data["image"].cuda()
    image_batch = image_batch.float().cuda()

    # ===================forward=====================
    output = model(image_batch)
    loss = criterion(output, image_batch)
    
    return loss

def train_on_device(cfg):

    min_valid_loss = np.inf

    learning_rate = 1e-3
    image_shape = (cfg.image_size, cfg.image_size, 3)
    n_channels=3

    print(image_shape)

    dataset = TrainDataset(path=cfg.train_data_dir, image_shape=image_shape) 
    train_dataset, val_dataset, threshold_dataset = split_data(dataset)
    
    train_dataset = DataLoader(train_dataset, batch_size=cfg.batch_size,shuffle=True)
    validate_dataset = DataLoader(val_dataset, batch_size=cfg.batch_size,shuffle=True)
    threshold_dataset = DataLoader(threshold_dataset, batch_size=cfg.batch_size,shuffle=True)

    model = Autoencoder(n_channels).cuda()
    criterion = SSIM(in_channels=n_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

    for epoch in range(cfg.n_epochs):
        total_train_loss = 0
        total_validation_loss = 0
        
        for i_train, train_batched in enumerate(train_dataset):          
            train_loss = train_one_step(model,criterion, optimizer,train_batched)
            total_train_loss += train_loss.data

        for i_validate, validate_batched in enumerate(validate_dataset):
            validation_loss = validate_one_step(model,criterion,validate_batched)
            total_validation_loss += validation_loss.data
            
        if min_valid_loss > total_validation_loss:
            print('Model saved - validation loss decreased - {:.4f} --> {:.4f}'.format(min_valid_loss, total_validation_loss))
            min_valid_loss = total_validation_loss
            torch.save(model.state_dict(), "model.pckl")
        
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, cfg.n_epochs, total_train_loss))

    threshold = get_threshold(threshold_dataset,model,cfg)
    print(threshold) 

# parse argument variables
cfg = Config().parse()

with torch.cuda.device(0):
    train_on_device(cfg)