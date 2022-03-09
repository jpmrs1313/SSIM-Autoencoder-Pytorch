import torch
from torch.utils.data import DataLoader
from config import Config
from data_loader import TrainDataset
from models import Autoencoder
from utils import *
import matplotlib.pyplot as plt

def test_on_device(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_shape = (cfg.image_size, cfg.image_size, 1)

    train_dataset = TrainDataset(path=cfg.train_data_dir, image_shape=image_shape) 
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size,shuffle=True)

    model = Autoencoder().to(device)
    model.load_state_dict(torch.load("model.pckl"))
    model.eval()

    results = []
    for i_batch, sample_batched in enumerate(dataloader):
        image_batch = sample_batched["image"].to(device)
        image_batch = image_batch.float().to(device)

        residual_maps = get_residual_map(image_batch,model)    
        results.extend(residual_maps)

    for result in results:
        image_pred = np.zeros((cfg.image_size,cfg.image_size), np.float32)
        image_pred[result >  0.45431541025638575 ] = 1

        x=image_pred
        plt.imshow(x, cmap="gray")
        plt.show()
        
    print(len(results))
    print(results[0].shape)
    
# parse argument variables
cfg = Config().parse()

with torch.cuda.device(0):
        test_on_device(cfg)