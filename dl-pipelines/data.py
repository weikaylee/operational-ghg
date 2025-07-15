import csv

import torch
from torchvision    import transforms

import cmutils
import cmutils.pytorch as cmtorch



def get_augment(mean, std, crop, ch4min, ch4max, s_min, s_max, u_min, u_max, 
                train=False, num_channels=1, norm="CMF", rotd=0.25, tdel=0.0125,
                rgbmin=0, rgbmax=20):
    """Define dataset preprocessing and augmentation"""

    preproc = [
        cmtorch.ClampMethaneTile(ch4min=ch4min, ch4max=ch4max,
                                 rgbmin=rgbmin, rgbmax=rgbmax),
        transforms.Normalize(mean, std)
    ]

    if num_channels == 2: 
        if norm == "CMF_SENS":
            preproc = [ 
                cmtorch.ClampTwoTile(ch4min=ch4min, ch4max=ch4max, min_two=s_min, max_two=s_max),
                transforms.Normalize(mean, std)
            ]
        elif norm == "CMF_UNCERT":
            preproc = [ 
                cmtorch.ClampTwoTile(ch4min=ch4min, ch4max=ch4max, min_two=u_min, max_two=u_max),
                transforms.Normalize(mean, std)
            ]
    elif num_channels == 3: 
        assert norm == "CMF_SENS_UNCERT"
        preproc = [ 
            cmtorch.ClampThreeTile(ch4min=ch4min, ch4max=ch4max,
                                 s_min=s_min, s_max=s_max, u_min=u_min, u_max=u_max),
            transforms.Normalize(mean, std)
        ]

    augment = []
    if train:
        rotang = [(-rotd,rotd)]+[(ra-rotd,ra+rotd) for ra in [90,180,270]]
        rottrn = [transforms.RandomAffine(rang,translate=(tdel,tdel))
                  for rang in rotang]
        rottrn = transforms.RandomChoice(rottrn)

        augment += [
            transforms.RandomApply(transforms=[rottrn],p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ]

    if crop:
        augment += [transforms.CenterCrop(crop)]

    preproc = transforms.Compose(preproc)
    augment = transforms.Compose(augment)

    return preproc, augment

def build_dataloader(csv_path,
                     root='/',
                     train=True,
                     num_channels=1, 
                     batch_size=8,
                     crop=None,
                     normmax=4000,
                     s_max=1.42, 
                     u_max=1039.1366, 
                     num_workers=4, 
                     norm=None):
    """
    Build a dataloader for segmentation tasks

    Parameters:
    - csv_path (str): Path to the CSV file containing image paths and labels.
    - root (str): Root directory for relative image paths. Defaults to '/' for absolute paths.
    - train (bool): Whether to apply training-specific augmentations. Defaults to True.
    - batch_size (int): Number of samples per batch. Defaults to 8.
    - crop (Optional[int]): Crop size for images, if applicable. Defaults to None.
    - normmax (float): Maximum value for normalization. Defaults to 4000.
    - num_workers (int): Number of subprocesses to use for data loading. Defaults to 4.

    Returns:
    - dataloader (DataLoader): A PyTorch DataLoader object containing the dataset.
    - lab_counts (list[int]): List with counts of positive and negative labels.
    """
    # Load data CSV
    datarows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            datarows.append(row)

    # Calculate loss weights to deal with imbalanced dataset
    all_labels = [1 if int(r[1]) == 1 else 0 for r in datarows]
    n = len(all_labels)
    npos = sum(all_labels)
    nneg = n - npos
    lab_counts = [npos, nneg]

    # Define transforms and dataset class
    if norm == "CMF_SENS_UNCERT": 
        mean, std = cmutils.CMF_SENS_UNCERT 
    elif norm == "CMF": 
        mean, std = cmutils.CMF 
    elif norm == "CMF_SENS": 
        mean, std = cmutils.CMF_SENS 
    elif norm == "CMF_UNCERT": 
        mean, std = cmutils.CMF_UNCERT
    else: 
        raise Exception(f"Undefined normalization: {norm}")
    
    # mean, std = 0, normmax # TODO why is this true (why does std = normax)? 
    ch4min = 0
    s_min = 0 
    u_min = 0
    ch4max = normmax

    # Define dataset
    dataset = cmtorch.SegmentDatasetCH4(
        root,
        datarows,
        *get_augment(mean, std, crop, ch4min, ch4max, s_min, s_max, u_min, u_max, 
                     train, num_channels, norm)
        # *get_augment(mean, std, crop, train=train, num_channels=num_channels,
        #              ch4min=ch4min, ch4max=ch4max, 
        #              s_min=s_min, s_max=s_max, 
        #              u_min=u_min, u_max=u_max)
    )

    # Define dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers
    )

    # Return dataloader and label counts for loss weighting
    return dataloader, lab_counts