from cProfile import label
import numpy
import torch
from torch.utils.data import DataLoader
from config import Config
from data_loader import TestDataset
from models import Autoencoder
from utils import *
import matplotlib.pyplot as plt

def test_on_device(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_shape = (cfg.image_size, cfg.image_size, 1)

    train_dataset = TestDataset(cfg.test_data_dir,cfg.mask_data_dir, image_shape=image_shape) 
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size,shuffle=True)

    model = Autoencoder().to(device)
    model.load_state_dict(torch.load("model.pckl"))
    model.eval()

    images, masks, predictions, labels = [], [], [], []
    for i_batch, sample_batched in enumerate(dataloader):
        image_batch = sample_batched["image"].cuda()
        image_batch = image_batch.float().cuda()

        mask_batch = sample_batched["mask"].cuda()
        mask_batch = mask_batch.float().cuda()

        label_batch = sample_batched["label"]

        residual_maps = get_residual_map(image_batch,model)    
        residual_maps = np.array(residual_maps) # Creating a tensor from a list of numpy.ndarrays is extremely slow. So i convert the list of numpy.ndarrays to a numpy.ndarrays
        results = torch.tensor(residual_maps)

        image_pred = np.zeros((len(results),cfg.image_size,cfg.image_size), np.float32)
        image_pred[results >  0.45431541025638575 ] = 1

        print
        images.extend(image_batch)
        masks.extend(mask_batch)
        labels.extend(label_batch)
        predictions.extend(image_pred)
    
    evaluate(masks,predictions,labels)
    
# parse argument variables
cfg = Config().parse()

with torch.cuda.device(0):
        test_on_device(cfg)