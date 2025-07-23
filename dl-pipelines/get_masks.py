import torch 
import os 
import shutil 
import csv 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rasterio as rio 
import argparse

from unet import DeepPaddedUNet
from data import build_dataloader

from tqdm           import tqdm
from PIL import Image

def get_masks(img_path, mask, target, output_dir, num_channels=1): 
    # Create a figure with 3 subplots (1 row, 3 columns)
    _, axs = plt.subplots(1, 5, figsize=(25, 5)) 

    # plot rgb
    base, _ = os.path.splitext(img_path)
    rgb_file = base + "_rgb.png"
    axs[0].imshow(mpimg.imread(rgb_file))
    axs[0].set_title('RGB')
    
    # plot predicted mask 
    axs[1].imshow(mask.squeeze().numpy(), cmap='gray')
    axs[1].set_title('Predicted Mask')
    
    # plot cmf 
    with rio.open(img_path) as src: 
        arr = src.read(1) 
        msk = src.read_masks(1) # maps pixels to 0 (non-valid) or 255 (valid)
        im1 = axs[2].imshow(arr * msk / 255, vmin=0, vmax=1500, cmap='viridis') 
        axs[2].set_title('CMF Retrieval')
        plt.colorbar(im1, ax=axs[2])
    
    uq_file = img_path 
    if num_channels != 3: 
    # plot sens, uncert
    # uq_product = batch['xpath'][i].replace('v1_accept20250530', 'v1_accept_cmfuq')
    # with rio.open(uq_product) as src: 
        uq_file = img_path.replace("v1_accept_sampled_cmf", "v1_accept_cmfuq")
    with rio.open(uq_file) as src: 
        band2 = src.read(2)  # sensitivity
        band2_msk = src.read_masks(2)
        band3 = src.read(3)  # uncertainty
        band3_msk = src.read_masks(3)
        
    im2 = axs[3].imshow(band2 * band2_msk / 255, vmin=0, vmax=2, cmap='viridis') 
    axs[3].set_title('Sensitivity')
    plt.colorbar(im2, ax=axs[3])
    im3 = axs[4].imshow(band3 * band3_msk / 255, vmin=0, vmax=4000, cmap='viridis') 
    axs[4].set_title('Uncertainty')
    plt.colorbar(im3, ax=axs[4])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save plot to a file
    output_dir_tp = os.path.join(output_dir, "tp")
    output_dir_tn = os.path.join(output_dir, "tn")
    output_dir_fp = os.path.join(output_dir, "fp")
    output_dir_fn = os.path.join(output_dir, "fn")
    
    pred_label = torch.max(mask) # 0 or 1
    true_label = torch.max(target) # 0 or 1
    
    filename = os.path.basename(img_path).split('.')[0] + "_mask.png"

    # true positive 
    save_path = os.path.join(output_dir_tp, filename)
    if pred_label == 0 and true_label == 0: # true negative 
        save_path = os.path.join(output_dir_tn, filename)
    elif pred_label == 1 and true_label == 0: # false positiive 
        save_path = os.path.join(output_dir_fp, filename)
    elif pred_label == 0 and true_label == 1: # false negative 
        save_path = os.path.join(output_dir_fn, filename)
    else: # true postive 
        save_path = os.path.join(output_dir_tp, filename)
                
    plt.savefig(save_path)
    plt.close()

def get_saliency_map(model, input_tensor, saliency_dir, img_path):
    """
    input_tensor: shape [1, C, H, W], requires_grad=True
    returns saliency map: [H, W]
    """
    print("ALSJD")
    print(input_tensor.shape)
    input_tensor.requires_grad_()
    
    output = model(input_tensor.unsqueeze(0))
    output = torch.sigmoid(output)
    
    score = output.sum()
    score.backward()
    
    saliency = input_tensor.grad.data.abs().squeeze(0)  # [C, H, W]
    saliency_map = saliency.max(dim=0)[0].cpu().numpy()  # max over channels → [H, W]
    
    # Save saliency map
    saliency_img_path = os.path.join(saliency_dir, os.path.basename(img_path).replace(".tif", "_saliency.png"))
    plt.imshow(saliency_map, cmap='hot')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(saliency_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def main(weights_path, valcsv, num_channels, output_dir):   
    weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/sampled_cmf/weights/399_seg_20250721_024454_v1_accept_sampled_cmf_train_absolute_gattaca_sampled_cmf_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # sampled cmf
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/199_seg_20250715_105036_train_absolute_analysis_cmfuq_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # three channel 
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/corrected_cmf_cont/199_seg_20250718_143904_v1_accept_corrected_cmf_train_absolute_gattaca_corrected_cmf_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # corrected_cmf_cont
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/cmfuq_lr/199_seg_20250718_141159_data_train_absolute_gattaca_cmfuq_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # cmfuq_lr
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/199_seg_20250711_164344_train_absolute_analysis_cmfuq_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # for the baseline model
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/thresholded_cmf_cont/199_seg_20250718_234404_v1_accept_thresholded_cmf_train_absolute_gattaca_thresholded_cmf_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # thresholded_cmf_cont
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/cmf_uncert/299_seg_20250717_135043_v1_accept_cmf_uncert_train_absolute_gattaca_cmf_uncert_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # cmf_uncert
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/cmf_sens/184_seg_20250717_223218_v1_accept_cmf_sens_train_absolute_gattaca_cmf_sens_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # cmf_sens
    # unetkws = dict(in_ch=2, num_classes=1, upsample_pad=False, pool="average") # default  
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/179_seg_20250702_143353_v1_accept20250530_tile_labs_noignore_nofidedge_train_utmzone_abspath_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_weights.pt" # decent snowflake
    unetkws = dict(in_ch=3, num_classes=1, upsample_pad=False, pool="average") # default besides in_ch 

    device = torch.device("cpu")
    model = DeepPaddedUNet(**unetkws).to(device) 
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # valcsv = "/Users/kwei/surf25/kaylee-surf25/data/test_absolute_local_cmf_sens.csv"
    valcsv = "/Users/kwei/surf25/kaylee-surf25/data/test_absolute_local_sampled_cmf.csv"
    # valcsv = "/Users/kwei/surf25/kaylee-surf25/data/test_absolute_local_cmfuq_subset.csv"
    dataloader, _ = build_dataloader(valcsv,
                                        root="/",
                                        train=False,
                                        num_channels=3,
                                        batch_size=8,
                                        normmax=4000, 
                                        norm="CMF_SENS_UNCERT")
    
    output_dir = "/Users/kwei/surf25/kaylee-surf25/figures/sampled_cmf"
    output_dir_tp = os.path.join(output_dir, "tp")
    output_dir_tn = os.path.join(output_dir, "tn")
    out_dir_fp = os.path.join(output_dir, "fp")
    out_dir_fn = os.path.join(output_dir, "fn")
    saliency_dir = os.path.join(output_dir, "saliency")

    output_dirs = [output_dir, output_dir_tn, output_dir_tp, out_dir_fp, out_dir_fn, saliency_dir]
    for dir in output_dirs: 
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
    
    fp_records = []
    fn_records = []
    num_tot_files = num_tp = num_tn = num_fp = num_fn = 0     
    
    for iter, batch in tqdm(enumerate(dataloader)):
        inputs = batch['x'].to(device) # change 
        # check range (clipped to 0, 1) and check dims 
        targets = batch['y'].to(device)        

        with torch.no_grad():
            batch_prob  = torch.sigmoid(model(inputs))
            masks = (batch_prob > 0.36).float() 

        # save each mask in batch
        for i in range(inputs.shape[0]):
            num_tot_files += 1
            img_path = batch['xpath'][i]
            mask = masks[i]
            target = targets[i]
            
            pred_label = torch.max(mask).item() # 0 or 1
            true_label = torch.max(target).item() # 0 or 1

            if pred_label == 1 and true_label == 0:
                # False Positive
                num_fp += 1
                fp_records.append({
                    "path": img_path,
                    "true_label": true_label,
                    "pred_label": pred_label
                })

            elif pred_label == 0 and true_label == 1:
                # False Negative
                num_fn += 1
                fn_records.append({
                    "path": img_path,
                    "true_label": true_label,
                    "pred_label": pred_label
                })
            elif pred_label == 0 and true_label == 0: 
                num_tn += 1 
            else: 
                num_tp += 1
                
            get_masks(img_path, mask, target, output_dir, num_channels=1)            
            # input_tensor = inputs[i].clone().detach().to(device).requires_grad_()
            # get_saliency_map(model, input_tensor, saliency_dir, img_path)

    fp_csv = os.path.join(output_dir, "false_positives.csv")
    fn_csv = os.path.join(output_dir, "false_negatives.csv")

    with open(fp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_label", "pred_label"])
        writer.writeheader()
        writer.writerows(fp_records)

    with open(fn_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_label", "pred_label"])
        writer.writeheader()
        writer.writerows(fn_records)
    
    assert(len(fp_records) == num_fp) 
    assert(len(fn_records) == num_fn)
    
    print(f"NUM TRUE POSITIVES: {num_tp}, {num_tp/num_tot_files}")
    print(f"NUM TRUE NEGATIVES: {num_tn}, {num_tn/num_tot_files}")
    print(f"NUM FALSE POSITIVES: {num_fp}, {num_fp/num_tot_files}")
    print(f"NUM FALSE NEGATIVES: {num_fn}, {num_fn/num_tot_files}")
    
    out_path = os.path.join(output_dir, "out.txt")
    with open(out_path, "w") as file:
        file.write(f"NUM TRUE POSITIVES: {num_tp}, {num_tp/num_tot_files}\n")
        file.write(f"NUM TRUE NEGATIVES: {num_tn}, {num_tn/num_tot_files}\n")
        file.write(f"NUM FALSE POSITIVES: {num_fp}, {num_fp/num_tot_files}\n")
        file.write(f"NUM FALSE NEGATIVES: {num_fn}, {num_fn/num_tot_files}\n")
                      
if __name__ == '__main__':     
    parser = argparse.ArgumentParser(description="Train a segmentation model on tiled methane data.")
    parser.add_argument('weight-file', help="Filepath of weights to load model with")
    parser.add_argument('valcsv', help="Filepath of validation set (with absolute paths) corresponding to model")
    parser.add_argument('--num-channels', help="Number of channels in input", type=int, default=1)
    parser.add_argument('--norm', choices=["CMF", "CMF_SENS_UNCERT", "CMF_SENS", "CMF_UNCERT", "CMF_MULTI"],
                                  default="CMF",
                                  help="Dataset statistic to use for normalization")
    parser.add_argument('--output-dir', help="Directory to dump outputs in", default=os.getwd())
    args = parser.parse_args() 
    
    weights_path = args.weight_file 
    valcsv = args.valcsv 
    num_channels = args.num_channels 
    norm = args.norm 
    output_dir = args.output_dir
    
    