import datetime
import json
from pathlib import Path



# Image Input/Output
# ----------------------------------------------------------------------------------------------
in_channels = 3
num_classes = 2
height = 400 #for PHR-CB experiment patch size = height = width
width = 400

# Training
# ----------------------------------------------------------------------------------------------
# mnet = fapnet, unet, ex_mnet*, dncnn, u2net, vnet, unet++, sm_unet, sm_linknet, sm_fpn, sm_pspnet*, kuc_vnet, kuc_unet3pp, kuc_r2unet,# kuc_unetpp*, kuc_restunet, kuc_tensnet*, kuc_swinnet, kuc_u2net, kuc_attunet, ad_unet, transformer
model_name = "kuc_r2unet"
batch_size = 32   #64
epochs = 500
# learning_rate = float 3e-4
val_plot_epoch = 10
augment = False
transfer_lr = False
gpu = "3"
trainOn = all  # ignore it

# Experiment Setup
# ----------------------------------------------------------------------------------------------
# regular/cls_balance/patchify/patchify_WOC
# cfr = regular, cfr_cb = cls_balance, phr = patchify, phr_cb = patchify_WOC
experiment = "phr_cb"

# Patchify (phr & phr_cb experiment)
# ----------------------------------------------------------------------------------------------
patchify = True
patch_class_balance = True # whether to use class balance while doing patchify
patch_size = 256 # height = width, anyone is suitable
stride = 64
p_train_dir = Path("json/train_patch_phr_cb_256.json")
p_valid_dir = Path("json/valid_patch_phr_cb_256.json")
p_test_dir = Path("json/test_patch_phr_cb_256.json")
p_eval_dir = Path("json/eval_patch_phr_cb_256.json")

# Dataset
# --------------------------------mask--------------------------------------------------------------
weights = False # False if cfr, True if cfr_cb
balance_weights = [1.8, 8.2]
dataset_dir = Path("/mnt/hdd2/mdsamiul/archive/road_segmentation/data")    # /mnt/hdd2/mdsamiul/archive/road_segmentation/data
root_dir = Path("/mnt/hdd2/mdsamiul/archive/road_segmentation")
train_size = 0.8
train_dir = "train.csv"
valid_dir = "valid.csv"
test_dir = "test.csv"
eval_dir = "eval.csv"

# Logger/Callbacks
# ----------------------------------------------------------------------------------------------
csv = True
val_pred_plot = True
lr = True
tensorboard = True
early_stop = False
checkpoint = True
patience = 300 # required for early_stopping, if accuracy does not change for 500 epochs, model will stop automatically

# Evaluation
# ----------------------------------------------------------------------------------------------
#load_model_name  = m.hdf5
load_model_name = "kuc_r2unet_ex_phr_cb_ep_500_24-Nov-22.hdf5"
load_model_dir = None #  If None, then by befault root_dir/model/model_name/load_model_name
evaluation = False # default evaluation value will not work
video_path = None    # If None, then by default root_dir/data/video_frame

# Prediction Plot
# ----------------------------------------------------------------------------------------------
plot_single = False # if True, then only index x_test image will plot # default plot_single  value will not work
index = -1 #170 # by default -1 means random image else specific index image provide by user


#  Create config path
# ----------------------------------------------------------------------------------------------
if patchify:
    height = patch_size
    width = patch_size

# Merge paths
train_dir = dataset_dir / train_dir
valid_dir = dataset_dir / valid_dir
test_dir = dataset_dir / test_dir
eval_dir = dataset_dir / eval_dir

p_train_dir = dataset_dir / p_train_dir
p_valid_dir = dataset_dir / p_valid_dir
p_test_dir = dataset_dir / p_test_dir
p_eval_dir = dataset_dir / p_eval_dir

# Create Callbacks paths
tensorboard_log_name = "{}_ex_{}_ep_{}".format(model_name, experiment, epochs)
tensorboard_log_dir = root_dir / "logs/tens_logger" / model_name

csv_log_name = "{}_ex_{}_ep_{}.csv".format(model_name, experiment, epochs)
csv_log_dir = root_dir / "logs/csv_logger" / model_name   

checkpoint_name = "{}_ex_{}_ep_{}.hdf5".format(model_name, experiment, epochs)
checkpoint_dir = root_dir / "logs/model" / model_name

# Create save model directory
if load_model_dir == None:
    load_model_dir = root_dir / "logs/model" / model_name 

# Create Evaluation directory
prediction_test_dir = root_dir / "logs/prediction/model_name/test" / experiment
prediction_eval_dir = root_dir / "logs/prediction/model_name/eval" / experiment
prediction_val_dir = root_dir / "logs/prediction/model_name/validation" / experiment

visualization_dir = root_dir / "logs/visualization/"