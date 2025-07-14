import torch 
import cmutils
import os
import shutil 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rasterio as rio 

from unet import DeepPaddedUNet
from data import build_dataloader

from tqdm           import tqdm
from PIL import Image

def main():     
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/analysis/apbest_seg_20250627_224605_train_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_weights.pt"
    weights_path = "/Users/kwei/surf25/kaylee-surf25/analysis/199_seg_20250711_164344_train_absolute_analysis_cmfuq_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt"
    unetkws = dict(in_ch=3, num_classes=1, upsample_pad=False, pool="average") # default besides in_ch 
    
    device = torch.device("cpu")
    model = DeepPaddedUNet(**unetkws).to(device) 
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    valcsv = "/Users/kwei/surf25/kaylee-surf25/data/test_absolute_local_cmfuq_subset.csv"
    dataloader, _ = build_dataloader(valcsv,
                                        root="/",
                                        train=False,
                                        usemulti=True,
                                        batch_size=8,
                                        normmax=4000, 
                                        norm="CMF_SENS_UNCERT")
    
    output_dir = "/Users/kwei/surf25/kaylee-surf25/figures/three_channel_masks_uq"
    output_dir_pos = os.path.join(output_dir, "pos")
    output_dir_neg = os.path.join(output_dir, "neg")
    
    output_dirs = [output_dir, output_dir_pos, output_dir_neg]
    for dir in output_dirs: 
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
    
    for iter, batch in tqdm(enumerate(dataloader)):
        inputs = batch['x'].to(device) # change 
        # check range (clipped to 0, 1) and check dims 
        targets = batch['y'].to(device)

        with torch.no_grad():
            batch_prob  = torch.sigmoid(model(inputs))
            masks = (batch_prob > 0.5).float() 

        # save each mask in batch
        for i in range(inputs.shape[0]):
            # Create a figure with 3 subplots (1 row, 3 columns)
            _, axs = plt.subplots(1, 5, figsize=(25, 5)) 

            # plot rgb
            base, _ = os.path.splitext(batch['xpath'][i])
            rgb_file = base + "_rgb.png"
            axs[0].imshow(mpimg.imread(rgb_file))
            axs[0].set_title('RGB')
            
            # plot predicted mask 
            axs[1].imshow(masks[i].squeeze().numpy(), cmap='gray')
            axs[1].set_title('Predicted Mask')
            
            # plot cmf 
            tif_file = batch['xpath'][i]
            with rio.open(tif_file) as src: 
                arr = src.read(1) 
                msk = src.read_masks(1) # maps pixels to 0 (non-valid) or 255 (valid)
                im1 = axs[2].imshow(arr * msk / 255, vmin=0, vmax=1500, cmap='viridis') 
                axs[2].set_title('CMF Retrieval')
                plt.colorbar(im1, ax=axs[2])
            
            # plot sens, uncert
            # uq_product = batch['xpath'][i].replace('v1_accept20250530', 'v1_accept_cmfuq_test')
            with rio.open(batch['xpath'][i]) as src: 
                band2 = src.read(2)  # sensitivity
                band2_msk = src.read_masks(2)
                band3 = src.read(3)  # uncertainty
                band3_msk = src.read_masks(3)
            im2 = axs[3].imshow(band2 * band2_msk / 255, cmap='viridis') 
            axs[3].set_title('Sensitivity')
            plt.colorbar(im2, ax=axs[3])
            im3 = axs[4].imshow(band3 * band3_msk / 255, cmap='viridis') 
            axs[4].set_title('Uncertainty')
            plt.colorbar(im3, ax=axs[4])

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Save plot to a file
            filename = os.path.basename(batch['xpath'][i]).split('.')[0] + "_mask.png"
            save_path = os.path.join(output_dir_neg, filename)
            if np.max(targets[i].numpy()): # if contains positive plume label
                save_path = os.path.join(output_dir_pos, filename)
            plt.savefig(save_path)
            plt.close()

            # Convert single-channel mask [1, H, W] to [H, W] and save
            # Image.fromarray(np.array(batch_prob[i].squeeze(0) * 255).astype('uint8')).convert("L").save(save_path)
            # vutils.save_image(batch_prob[i], save_path)
                      
if __name__ == '__main__': 
    main()
