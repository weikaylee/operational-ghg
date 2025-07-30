import torch
import os
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rasterio as rio
import argparse
import sys 

from unet import DeepPaddedUNet
from data import build_dataloader

from tqdm import tqdm

def get_stitched(img_path, mask, salience, output_dir, threshold, confidence, pred_label):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].remove()
        
    # plot predicted mask
    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Predicted Mask")

    # plot salience 
    im = axs[2].imshow(salience, vmin=0, vmax=1) 
    axs[2].set_title("Predicted Salience")
    plt.colorbar(im, ax=axs[2])
    
    # plot cmf (if decent snowflake) or sampled cmf (if sampled cmf)
    if "sampled_cmf" in img_path: 
        with rio.open(img_path) as src: 
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(3).astype(np.float32)

            # Step 1: Clip values to [0, 1500]
            r = np.clip(r, 0, 1500) / 1500
            g = np.clip(g, 0, 1500) / 1500
            b = np.clip(b, 0, 1500) / 1500

            # Step 3: Stack into RGB image
            rgb_image = np.stack([r, g, b], axis=-1)
            axs[3].imshow(rgb_image)
            axs[3].set_title("Sampled CMF RGB")
    else: 
        with rio.open(img_path) as src:
            arr = src.read(1)
            msk = src.read_masks(1)  # maps pixels to 0 (non-valid) or 255 (valid)
            im = axs[3].imshow(arr * msk / 255, vmin=0, vmax=1500, cmap="viridis")
            axs[3].set_title("CMF Retrieval")     
        plt.colorbar(im, ax=axs[3])

    # axs[4].imshow(band2 * band2_msk / 255, vmin=0, vmax=1.42, cmap="viridis")
    # axs[4].set_title("Sensitivity")
    # axs[5].imshow(band3 * band3_msk / 255, vmin=0, vmax=1039.1366, cmap="viridis")
    # axs[5].set_title("Uncertainty")
    
    # remove axs ticks 
    for ax in axs:
        ax.axis('off')
    
    # save stitched, salience, and binary separately 
    filename = os.path.basename(img_path).split(".")[0] 
    save_path_stitched = os.path.join(output_dir, "stitched")
    os.makedirs(save_path_stitched, exist_ok=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.87])
    if "sampled_cmf" in img_path: 
        # fig.text(0.05, 0.5, f"Sampled CMF\nSeg threshold={round(threshold, 2)}", va='center', ha='right',
        # fontsize=16, rotation=90, fontweight='bold')
        fig.suptitle(f"Sampled CMF\nPred label={pred_label}, confidence={confidence:.2f}, seg threshold={threshold:.2f}", fontweight="bold")
    else: 
        # fig.text(0.05, 0.5, f"Decent Snowflake\nSeg threshold={round(threshold, 2)}", va='center', ha='right',
        # fontsize=16, rotation=90, fontweight='bold')
        fig.suptitle(f"Decent Snowflake\nPred label={pred_label}, confidence={confidence:.2f}, seg threshold={threshold:.2f}", fontweight="bold")
    plt.savefig(os.path.join(save_path_stitched, filename + ".pdf"), dpi=300)
    print(f"Saving {filename}...")
    plt.close()

def get_rgb_true_uq(img_path, true_mask, uq_path, true_label): 
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # plot rgb
    base, _ = os.path.splitext(img_path)
    rgb_file = base + "_rgb.png"
    axs[0].imshow(mpimg.imread(rgb_file))
    axs[0].set_title("RGB")

    # plot true mask 
    axs[1].imshow(true_mask, cmap="gray")
    axs[1].set_title("True Mask")
    
    # plot uq 
    with rio.open(uq_path) as src:
        arr = src.read(2)
        msk = src.read_masks(2)  # maps pixels to 0 (non-valid) or 255 (valid)
        im = axs[2].imshow(arr * msk / 255, vmin=0, vmax=1.42, cmap="viridis")
        axs[2].set_title("Sensitivity")
        plt.colorbar(im, ax=axs[2])
        
        arr = src.read(3) 
        msk = src.read_masks(3) 
        im = axs[3].imshow(arr * msk / 255, vmin=0, vmax=1039.1366, cmap="viridis")
        axs[3].set_title("Uncertainty")
        plt.colorbar(im, ax=axs[3])
    
    filename = os.path.basename(img_path).split(".")[0] 
    save_path_info = os.path.join(output_dir, "info")
    os.makedirs(save_path_info, exist_ok=True)
    
    for ax in axs:
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.87])
    # fig.text(0.05, 0.5, f"{filename}\nTrue label={true_label}, Pred label={pred_label}, Confidence={round(confidence, 2)}", va='center', ha='right',
    #     fontsize=16, rotation=90, fontweight='bold')
    fig.suptitle(f"{filename}\nTrue label={int(true_label)}", fontweight="bold")
    plt.savefig(os.path.join(save_path_info, filename + ".pdf"), dpi=300)
    print(f"Saving {filename}...")
    plt.close()

def main(weights_path, valcsv, num_channels, norm, output_dir, threshold):
    unetkws = dict(
        in_ch=num_channels, num_classes=1, upsample_pad=False, pool="average"
    )
    device = torch.device("cpu")
    model = DeepPaddedUNet(**unetkws).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    dataloader, _ = build_dataloader(
        valcsv,
        root="/",
        train=False,
        num_channels=num_channels,
        batch_size=8,
        normmax=4000,
        norm=norm,
    )

    if os.path.exists(output_dir):
        while True:
            response = input(f"Warning: The directory '{output_dir}' already exists. Continue? (yes/no): ").strip().lower()
            if response == 'yes':
                break  # proceed
            elif response == 'no':
                print("Exiting program.")
                sys.exit(1)
            else:
                print("Please enter 'yes' or 'no'.")
    else: 
        os.makedirs(output_dir)

    num_tot_files = num_tp = num_tn = num_fp = num_fn = 0

    predictions_csv = os.path.join(output_dir, "predictions.csv")
    mispredictions_csv = os.path.join(output_dir, "mispredictions.csv")
    fieldnames=["path", "true_label", "pred_label", "confidence"]

    with open(predictions_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
    with open(mispredictions_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
    # Open predictions CSV in append mode to write image-by-image
    with open(predictions_csv, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Open mispredictions CSV in append mode to write image-by-image
        with open(mispredictions_csv, mode="a", newline="") as mf:
            mispredictions_writer = csv.DictWriter(mf, fieldnames=fieldnames)

            # Running the loop over the batch
            for _, batch in tqdm(enumerate(dataloader)):
                inputs = batch["x"].to(device)
                targets = batch["y"].to(device)

                with torch.no_grad():
                    batch_prob = torch.sigmoid(model(inputs))
                    masks = (batch_prob > threshold).float()

                # save each mask in batch
                for i in range(inputs.shape[0]):
                    num_tot_files += 1
                    img_path = batch["xpath"][i]
                    mask = masks[i].squeeze().numpy()
                    salience = batch_prob[i].squeeze().numpy()
                    target = targets[i].squeeze().numpy()
                    
                    base = os.path.basename(img_path) 
                    salience_path_1 = os.path.join("/data/MLIA_active_data/data_CH4/models/EMIT/v1_accept20250530/pred_out/decent-snowflake-1/v1_accept20250530/tile_labs_noignore_nofidedge_abspath_DeepUCNet_predictions/bg", base)
                    salience_path_2 = os.path.join("/data/MLIA_active_data/data_CH4/models/EMIT/v1_accept20250530/pred_out/decent-snowflake-1/v1_accept20250530/tile_labs_noignore_nofidedge_abspath_DeepUCNet_predictions/pos", base)
                    
                    # get labels from salience mask for decent snowflake 
                    if "v1_accept20250530" in img_path: 
                        if os.path.exists(salience_path_1):
                            with rio.open(salience_path_1) as src:
                                salience = src.read(1)
                                mask = (salience > threshold).astype(int)
                        elif os.path.exists(salience_path_2): 
                            with rio.open(salience_path_2) as src:
                                salience = src.read(1)
                                mask = (salience > threshold).astype(int)
                        else: 
                            print("Salience file does not exist")
                            sys.exit(1)

                    # TODO hardcoded lol
                    parts = img_path.split(os.sep)
                    parts[-4] = "v1_accept_cmfuq"
                    uq_path = os.sep.join(parts)

                    pred_label = np.max(mask)  # 0 or 1
                    true_label = np.max(target)  # 0 or 1
                    confidence = np.max(salience) # max probablity among all pixels 

                    if pred_label == 1 and true_label == 0: # false positive 
                        num_fp += 1
                    elif pred_label == 0 and true_label == 1: # false negative 
                        num_fn += 1
                    elif pred_label == 0 and true_label == 0: # true negative 
                        num_tn += 1
                    else: # true positive 
                        num_tp += 1

                    info = {
                        "path": img_path,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "confidence": confidence
                    }

                    writer.writerow(info)
                    if pred_label != true_label: 
                        mispredictions_writer.writerow(info)

                    # get_rgb_true_uq(img_path, target, uq_path, true_label)
                    get_stitched(img_path, mask, salience, output_dir, threshold, confidence, pred_label)
                    
    out_path = os.path.join(output_dir, "stats.txt")
    precision = num_tp / (num_tp + num_fp)
    recall = num_tp / (num_tp + num_fn)
    f1 = 2 * precision * recall / (precision + recall)
    with open(out_path, "w") as file:
        file.write(f"True positive rate: {num_tp/num_tot_files} ({num_tp}/{num_tot_files})\n")
        file.write(f"True negative rate: {num_tn/num_tot_files} ({num_tn}/{num_tot_files})\n")
        file.write(f"False positive rate: {num_fp/num_tot_files} ({num_fp}/{num_tot_files})\n")
        file.write(f"False negative rate: {num_fn/num_tot_files} ({num_fn}/{num_tot_files})\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1: {f1}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a segmentation model on tiled methane data."
    )
    
    parser.add_argument("weights", help="Filepath of weights to load model with")
    parser.add_argument(
        "valcsv",
        help="Filepath of validation set (with absolute paths) corresponding to model",
    )
    parser.add_argument(
        "--num-channels", help="Number of channels in input", type=int, default=1
    )
    parser.add_argument(
        "--norm",
        choices=["CMF", "CMF_SENS_UNCERT", "CMF_SENS", "CMF_UNCERT", "CMF_MULTI"],
        default="CMF",
        help="Dataset statistic to use for normalization",
    )
    parser.add_argument(
        "--output-dir", help="Directory to dump outputs in", default=os.getcwd()
    )
    parser.add_argument("--threshold", help="Segmentation threshold", type=float, default=0.5)
    args = parser.parse_args()

    weights_path = args.weights
    valcsv = args.valcsv
    num_channels = args.num_channels
    norm = args.norm
    output_dir = args.output_dir
    threshold = args.threshold 
    
    main(weights_path, valcsv, num_channels, norm, output_dir, threshold)

    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/sampled_cmf/weights/399_seg_20250721_024454_v1_accept_sampled_cmf_train_absolute_gattaca_sampled_cmf_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # sampled cmf
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/199_seg_20250715_105036_train_absolute_analysis_cmfuq_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # three channel
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/corrected_cmf_cont/199_seg_20250718_143904_v1_accept_corrected_cmf_train_absolute_gattaca_corrected_cmf_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # corrected_cmf_cont
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/cmfuq_lr/199_seg_20250718_141159_data_train_absolute_gattaca_cmfuq_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # cmfuq_lr
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/199_seg_20250711_164344_train_absolute_analysis_cmfuq_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # for the baseline model
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/thresholded_cmf_cont/199_seg_20250718_234404_v1_accept_thresholded_cmf_train_absolute_gattaca_thresholded_cmf_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # thresholded_cmf_cont
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/cmf_uncert/299_seg_20250717_135043_v1_accept_cmf_uncert_train_absolute_gattaca_cmf_uncert_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # cmf_uncert
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/cmf_sens/184_seg_20250717_223218_v1_accept_cmf_sens_train_absolute_gattaca_cmf_sens_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_cmfuq_weights.pt" # cmf_sens
    # weights_path = "/Users/kwei/surf25/kaylee-surf25/data/weights/179_seg_20250702_143353_v1_accept20250530_tile_labs_noignore_nofidedge_train_utmzone_abspath_sigmoid_unetdeep_averagepool_ch4_crop256_clsaggmax_weights.pt" # decent snowflake

    # valcsv = "/Users/kwei/surf25/kaylee-surf25/data/test_absolute_local_cmf_sens.csv"
    # valcsv = "/Users/kwei/surf25/kaylee-surf25/data/test_absolute_local_sampled_cmf.csv"
    # valcsv = "/Users/kwei/surf25/kaylee-surf25/data/test_absolute_local_cmfuq_subset.csv"
