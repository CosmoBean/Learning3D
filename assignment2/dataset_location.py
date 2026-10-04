# specify the root location where u downloaded the dataset
import os
# R2N2_ROOT / R2N2_FULL=1 let concurrent runs pick a dataset without editing this file
root_location = os.environ.get("R2N2_ROOT", "data")
use_full_dataset = os.environ.get("R2N2_FULL") == "1"
dataset_name = (
    "r2n2_shapenet_dataset_full" if use_full_dataset else "r2n2_shapenet_dataset"
)

R2N2_PATH = f"{root_location}/{dataset_name}/r2n2"
SHAPENET_PATH = f"{root_location}/{dataset_name}/shapenet"

if use_full_dataset:
    SPLITS_PATH = f"{root_location}/{dataset_name}/split_3c.json"  # split file contains data entry for 3 classes
else:
    SPLITS_PATH = f"{root_location}/{dataset_name}/split_03001627.json"  # split file contains data entry for 03001627 class
