import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import sklearn.metrics as metrics
from skimage.segmentation import join_segmentations, mark_boundaries

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

def plot_images(x,y_true,y_pred):
    x=x.permute(1,2,0)
    x=x.cpu().data.numpy()
    x=np.squeeze(x)

    y_true=y_true.permute(1,2,0)
    y_true=y_true.cpu().data.numpy()
    y_true=np.squeeze(y_true)
    y_true=y_true.astype(int)

    y_pred=y_pred.astype(int)

    segj = join_segmentations(y_true,y_pred)
    boundaries = mark_boundaries(x,y_pred)

    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 4
    fig.add_subplot(rows, columns, 1)

    plt.imshow(y_pred)
    plt.title("prediction")
    plt.axis("off")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(y_true)
    plt.title("ground_truth")
    plt.axis("off")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(segj)
    plt.title("join")
    plt.axis("off")

    fig.add_subplot(rows, columns, 4)
    plt.imshow(boundaries)
    plt.title("join")
    plt.axis("off")

    plt.show()

def evaluate(y_true,y_pred,labels):
   
    print("Defects detection results:")
    labels_pred=[]

    for y_p in y_pred:
        if(not np.any(y_p)): 
            labels_pred.append(0)
        else: 
            labels_pred.append(1)
    
    CM = metrics.confusion_matrix(labels, labels_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    print(CM)
    print("classification of good samples: ", TN/(TN+FP))
    print("classification of defective samples: ", TP/(TP+FN))
    
    print("Defects Segmentation results:")
    
    y_true_np_list = []
    for y in y_true:
        y_true_np_list.append(y.cpu().data.numpy())

    y_true_np_list=np.squeeze(y_true_np_list)

    y_true_np = np.array(y_true_np_list)
    y_pred = np.array(y_pred)

    y_pred = y_pred.flatten()
    y_true_np = y_true_np.flatten()

    print("IOU "+ str(metrics.jaccard_score(y_pred,y_true_np)))
    print("FScore " + str(metrics.f1_score(y_pred,y_true_np)))
    print("AUC PR " + str(metrics.average_precision_score(y_pred,y_true_np)))
    print("AUC ROC " + str(metrics.roc_auc_score(y_pred,y_true_np)))
