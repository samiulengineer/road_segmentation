import os
import time
import argparse
from loss import *
from config import *
from metrics import *
from tensorflow import keras
from dataset import get_test_dataloader
from tensorflow.keras.models import load_model
from utils import create_paths, patch_show_predictions, frame_to_video


# Parsing variable
# ----------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--load_model_name")
parser.add_argument("--plot_single")
parser.add_argument("--index", type=int)
parser.add_argument("--experiment")
parser.add_argument("--gpu")
parser.add_argument("--evaluation")
parser.add_argument("--video_path")
args = parser.parse_args()

if args.plot_single == 'True':
    args.plot_single = True
else:
    args.plot_single = False
    
if args.evaluation == 'True':
    args.evaluation = True
else:
    args.evaluation = False

t0 = time.time()

# Set up test configaration
# ----------------------------------------------------------------------------------------------
if evaluation:
    create_paths(eval = True)
else:
    create_paths(test = True)



# setup gpu
# ----------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load Model
# ----------------------------------------------------------------------------------------------
print("Loading model {} from {}".format(load_model_name, load_model_dir))
# with strategy.scope(): # if multiple GPU is required
model = load_model((load_model_dir / load_model_name), compile=False)


# Dataset
# ----------------------------------------------------------------------------------------------
test_dataset = get_test_dataloader()

# Prediction Plot
# ----------------------------------------------------------------------------------------------
print("Saving test/evaluation predictions...")
print("call patch_show_predictions")
patch_show_predictions(test_dataset, model)


# Evaluation Score
# ----------------------------------------------------------------------------------------------
if not evaluation:
    metrics = list(get_metrics().values())
    adam = keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(optimizer=adam, loss=focal_loss(), metrics=metrics)
    model.evaluate(test_dataset)

# Frame to video
# ----------------------------------------------------------------------------------------------
if video_path != 'None':
    fname = dataset_dir + 'prediction.avi'
    frame_to_video(fname, fps=30)


print("training time sec: {}".format((time.time()-t0)))