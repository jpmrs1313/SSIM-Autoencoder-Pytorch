import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import sklearn.metrics as metrics
from skimage.segmentation import join_segmentations, mark_boundaries, clear_border
from skimage.morphology import disk, opening
from skimage.measure import label, regionprops

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

def get_threshold(dataset,model,cfg):
    total_rec= []
    for i_batch, batch in enumerate(dataset): 
            residual_maps = get_residual_map(batch['image'],model,cfg)
            total_rec.extend(residual_maps)
         
    total_rec = np.array(total_rec)

    return float(np.percentile(total_rec, [99]))

def get_residual_map(batch, model,cfg):
    residual_maps=[]
    batch=batch.float().cuda()
    results =  model(batch)

    for image,result in zip(batch,results):
        if cfg.grayscale=="True":
            image = image.cpu().detach().numpy()
            image = np.squeeze(image)
            result = result.cpu().detach().numpy()
            result = np.squeeze(result)
            residual_map = 1 - ssim(image,result, win_size=11, full=True)[1]
        else:
            image=image.permute(1,2,0)
            image = image.cpu().detach().numpy()
            result=result.permute(1,2,0)
            result = result.cpu().detach().numpy()
            residual_map = ssim(image, result, win_size=11, full=True, multichannel=True)[1]
            residual_map = 1 - np.mean(residual_map, axis=2)
        residual_maps.append(residual_map)

    return results, residual_maps

def plot_images(image,prediction,true_mask,pred_mask, residual):

    image=image.transpose(1,2,0)
    image=np.squeeze(image)

    true_mask=true_mask.transpose(1,2,0)
    true_mask=np.squeeze(true_mask)
    
    true_mask=true_mask.astype(int)
    pred_mask=pred_mask.astype(int)

    true_boundary = mark_boundaries(image,true_mask)
    pred_boundary = mark_boundaries(image,pred_mask)

    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 3
    
    fig.add_subplot(rows, columns, 1)
    plt.imshow(true_boundary)
    plt.title("Image")
    plt.axis("off")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(residual)
    plt.title("Anomaly Map")
    plt.axis("off")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(pred_boundary)
    plt.title("Defects Localization")
    plt.axis("off")

    plt.show()

def compute_pro(super_mask,gt_mask):
    max_step = 1000
    expect_fpr = 0.3  # default 30%
    max_th = super_mask.max()
    min_th = super_mask.min()
    delta = (max_th - min_th) / max_step
    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(super_mask, dtype=np.bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[super_mask <= thred] = 0
        binary_score_maps[super_mask >  thred] = 1
        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
        for i in range(len(binary_score_maps)):    # for i th image
            # pro (per region level)
            label_map = label(gt_mask[i], connectivity=2)
            props = regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_mask = prop.filled_image    # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
            if gt_mask[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        gt_masks_neg = ~gt_mask
        fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)
    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)
    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)
    # best per image iou
    best_miou = ious_mean.max()
    #print(f"Best IOU: {best_miou:.4f}")
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected =  (fprs_selected - fprs_selected.min()) / (fprs_selected.max() - fprs_selected.min())
    pros_mean_selected = pros_mean[idx]    
    return metrics.auc(fprs_selected, pros_mean_selected)

def evaluate(y_trues,residuals,labels, threshold,cfg):
    residuals=np.array(residuals)

    score_label = np.max(residuals, axis=(1, 2))
    gt_label = np.asarray(labels, dtype=np.bool)
    det_roc_auc = metrics.roc_auc_score(gt_label, score_label)
    
    gt_mask = np.squeeze(np.asarray(y_trues, dtype=np.bool), axis=1)
    seg_roc_auc = metrics.roc_auc_score(gt_mask.flatten(), residuals.flatten())

    print("Detection AUC: " + str(det_roc_auc))
    print("Segmentation AUC: " + str(seg_roc_auc))

    pro = compute_pro(residuals,gt_mask)
    
    print("Segmentation PRO: " + str(pro) )
  
   


  
