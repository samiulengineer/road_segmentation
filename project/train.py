import os
import time
import argparse
from loss import *
from config import *
from metrics import get_metrics
import tensorflow_addons as tfa
from dataset import get_train_val_dataloader
from tensorflow.keras.models import load_model
from model import get_model, get_model_transfer_lr
from utils import SelectCallbacks, get_config_yaml, create_paths

tf.config.optimizer.set_jit("True")


# Parsing variable ctrl + /
# ----------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir")
parser.add_argument("--dataset_dir")
parser.add_argument("--model_name")
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--experiment")
parser.add_argument("--gpu")

args = parser.parse_args()


# Set up train configaration
# ----------------------------------------------------------------------------------------------
# config = get_config_yaml('project/config.yaml', vars(args))
create_paths(test = False)


# setup gpu
# ----------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Print Experimental Setup before Training
# ----------------------------------------------------------------------------------------------
print("Model = {}".format(model_name))
print("Epochs = {}".format(epochs))
print("Batch Size = {}".format(batch_size))
print("Preprocessed Data = {}".format(os.path.exists(train_dir)))
print("Class Weigth = {}".format(str(weights)))
print("Experiment = {}".format(str(experiment)))

# Dataset
# ----------------------------------------------------------------------------------------------
train_dataset, val_dataset = get_train_val_dataloader()


# Metrics
# ----------------------------------------------------------------------------------------------
metrics = list(get_metrics().values())  # [list] required for new model
custom_obj = get_metrics() # [dictionary] required for transfer learning & fine tuning

# Optimizer
# ----------------------------------------------------------------------------------------------
learning_rate = 0.001
weight_decay = 0.0001
adam = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay)

# Loss Function
# ----------------------------------------------------------------------------------------------
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) # required for new model
custom_obj['loss'] = focal_loss() # required for transfer learning/fine-tuning

# Compile
# ----------------------------------------------------------------------------------------------
# transfer learning
if (os.path.exists(load_model_dir / load_model_name)) and transfer_lr:
    print("Build model for transfer learning..")
    # load model and compile
    model = load_model((load_model_dir / load_model_name), custom_objects=custom_obj, compile=True)

    model = get_model_transfer_lr(model, num_classes)
    model.compile(optimizer=adam, loss=loss, metrics=metrics)

else:
    # fine-tuning
    if (os.path.exists(load_model_dir / load_model_name)):
        print("Resume training from model checkpoint {}...".format(load_model_name))
        # load model and compile
        model = load_model((load_model_dir / load_model_name), custom_objects=custom_obj, compile=True)

    # new model
    else:
        model = get_model()
        model.compile(optimizer=adam, loss=loss, metrics=metrics)

# Callbacks
# ----------------------------------------------------------------------------------------------
loggers = SelectCallbacks(val_dataset, model)
model.summary()

# Fit
# ----------------------------------------------------------------------------------------------
t0 = time.time()
history = model.fit(train_dataset,
                    verbose = 1,
                    epochs = epochs,
                    validation_data = val_dataset,
                    shuffle = False,
                    callbacks = loggers.get_callbacks(val_dataset, model),
                    )


print("training time minute: {}".format((time.time()-t0)/60))
