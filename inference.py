import torch
from torch.utils.data import DataLoader
from config import Config
from data_loader import TestDataset
from models import Autoencoder
from utils import *
import time 

def inference_on_device(cfg):
    image_shape = (cfg.image_size, cfg.image_size, 3)
    n_channels=3

    mask_shape =  (cfg.image_size, cfg.image_size, 1)

    test_dataset = TestDataset(cfg.test_data_dir,cfg.mask_data_dir, image_shape=image_shape, mask_shape=mask_shape) 
    dataloader = DataLoader(test_dataset, batch_size=32,shuffle=True)

    model = Autoencoder(n_channels=n_channels).cuda()
    model.load_state_dict(torch.load("model.pckl"))
    model.eval()

    start = time.time()
    residuals = []
    for i_batch, sample_batched in enumerate(dataloader):
        image_batch = sample_batched["image"].cuda()
        image_batch = image_batch.float().cuda()

        preds,residual_maps = get_residual_map(image_batch,model,cfg)   
        residual_maps = np.array(residual_maps) # Creating a tensor from a list of numpy.ndarrays is extremely slow. So i convert the list of numpy.ndarrays to a numpy.ndarrays
        residuals.extend(residual_maps)
    
    fps = len(dataloader.dataset) / (time.time() - start)
    print('{:.2f} fps'.format(fps))

cfg = Config().parse()
cfg.test_data_dir=r"C:\Users\jpmrs\OneDrive\Desktop\Dissertação\code\data\mvtec\leather\test"
cfg.mask_data_dir=r"C:\Users\jpmrs\OneDrive\Desktop\Dissertação\code\data\mvtec\leather\ground_truth"

with torch.cuda.device(0):
        inference_on_device(cfg)