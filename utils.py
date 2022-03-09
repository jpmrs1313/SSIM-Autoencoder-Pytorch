import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def split_data(dataset):
    """Split data in train 70%, validation 15% and test 15%

    Parameters
    -----------
    dataset:  
    Returns
    -----------
    train_dataset, val_dataset, test_dataset
    """
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    test_size = int(0.5 * len(test_dataset))
    val_size = len(test_dataset) - test_size

    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def get_threshold(dataset,model):
    total_rec= []
    for i_batch, batch in enumerate(dataset): 
            residual_maps = get_residual_map(batch['image'],model)
            total_rec.extend(residual_maps)
         
    total_rec = np.array(total_rec)

    return float(np.percentile(total_rec, [99]))

def get_residual_map(batch, model):
    residual_maps=[]
    batch=batch.float().cuda()
    results =  model(batch)

    for image,result in zip(batch,results):
        image = image.cpu().detach().numpy()
        image = np.squeeze(image)
        result = result.cpu().detach().numpy()
        result = np.squeeze(result)

        residual_map = 1 - ssim(image,result, win_size=11, full=True)[1]
        residual_maps.append(residual_map)

    return residual_maps