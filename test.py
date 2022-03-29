import torch
from torch.utils.data import DataLoader
from config import Config
from data_loader import TestDataset
from models import Autoencoder
from utils import *

def test_on_device(cfg):
    if cfg.grayscale=="True":
        image_shape = (cfg.image_size, cfg.image_size, 1)
        n_channels=1
    else:
        image_shape = (cfg.image_size, cfg.image_size, 3)
        n_channels=3

    mask_shape =  (cfg.image_size, cfg.image_size, 1)

    test_dataset = TestDataset(cfg.test_data_dir,cfg.mask_data_dir, image_shape=image_shape, mask_shape=mask_shape) 
    dataloader = DataLoader(test_dataset, batch_size=1,shuffle=True)

    model = Autoencoder(n_channels=n_channels).cuda()
    model.load_state_dict(torch.load("model_color.pckl"))
    model.eval()

    images, true_masks, predictions, pred_masks, residuals, labels = [], [], [], [], [], []
    for i_batch, sample_batched in enumerate(dataloader):
        image_batch = sample_batched["image"].cuda()
        image_batch = image_batch.float().cuda()
        mask_batch = sample_batched["mask"].cuda()
        mask_batch = mask_batch.float().cuda()

        label_batch = sample_batched["label"]

        preds,residual_maps = get_residual_map(image_batch,model,cfg)   
        residual_maps = np.array(residual_maps) # Creating a tensor from a list of numpy.ndarrays is extremely slow. So i convert the list of numpy.ndarrays to a numpy.ndarrays

        threshold=0.4798461836576462

        images.extend(image_batch.cpu().data.numpy())
        true_masks.extend(mask_batch.cpu().data.numpy())
        labels.extend(label_batch.numpy())
        predictions.extend(preds.cpu().data.numpy())
        residuals.extend(residual_maps)
      
    evaluate(true_masks,residuals,labels, threshold,cfg)

    for residual in residuals:
        binary_score_maps = np.zeros_like(residual, dtype=np.bool)
        binary_score_maps[residual >  threshold] = 1
        binary_score_maps=clear_border(binary_score_maps,3)
        pred_masks.append(binary_score_maps)

    for image,prediction,true_mask,pred_mask, residual in zip(images,predictions,true_masks, pred_masks, residuals):
        plot_images(image,prediction, true_mask, pred_mask, residual)


cfg = Config().parse()

with torch.cuda.device(0):
        test_on_device(cfg)