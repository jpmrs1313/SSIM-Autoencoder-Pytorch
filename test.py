import torch
from torch.utils.data import DataLoader
from config import Config
from data_loader import TestDataset
from models import Autoencoder
from utils import *
 
def test_on_device(cfg):
    image_shape = (cfg.image_size, cfg.image_size, 3)
    n_channels=3

    mask_shape =  (cfg.image_size, cfg.image_size, 1)

    test_dataset = TestDataset(cfg.test_data_dir,cfg.mask_data_dir, image_shape=image_shape, mask_shape=mask_shape) 
    dataloader = DataLoader(test_dataset, batch_size=32,shuffle=True)

    model = Autoencoder(n_channels=n_channels).cuda()
    model.load_state_dict(torch.load("model.pckl"))
    model.eval()

    threshold=0.5898461836576462

    images, true_masks, predictions, pred_masks, residuals, labels = [], [], [], [], [], []
    for i_batch, sample_batched in enumerate(dataloader):

        image_batch = sample_batched["image"].cuda()
        image_batch = image_batch.float().cuda()
        mask_batch = sample_batched["mask"].cuda()
        mask_batch = mask_batch.float().cuda()

        label_batch = sample_batched["label"]

        preds,residual_maps = get_residual_map(image_batch,model,cfg)   
        residual_maps = np.array(residual_maps) # Creating a tensor from a list of numpy.ndarrays is extremely slow. So i convert the list of numpy.ndarrays to a numpy.ndarrays

        images.extend(image_batch.cpu().data.numpy())
        true_masks.extend(mask_batch.cpu().data.numpy())
        labels.extend(label_batch.numpy())
        predictions.extend(preds.cpu().data.numpy())
        residuals.extend(residual_maps)

    evaluate(true_masks,residuals,labels)
    visualize(images, true_masks ,residuals,threshold)

cfg = Config().parse()
cfg.test_data_dir=r"C:\Users\jpmrs\OneDrive\Desktop\Dissertação\code\data\mvtec\leather\test"
cfg.mask_data_dir=r"C:\Users\jpmrs\OneDrive\Desktop\Dissertação\code\data\mvtec\leather\ground_truth"
cfg.grayscale = "False"
with torch.cuda.device(0):
        test_on_device(cfg)