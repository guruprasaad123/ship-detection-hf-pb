import torch
import os

BATCH_SIZE = 1 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 105 # number of epochs to train for
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 1

# training images and XML files directory
TRAIN_DIR = os.path.join('..','input','train')

# validation images and XML files directory
TEST_DIR = os.path.join('..','input','test')

# classes: 0 index is reserved for background
CLASSES = [
    'None',
    'ship'
]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = os.path.join('..','output')

SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs
SAVE_FOR_EVERY_EPOCH = 5 # save model for every N epochs 

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = os.path.join('..','output')

# location to save the logs
LOG_DIR = os.path.join('..','logs')

# 'fasterrcnn_resnet18','fasterrcnn_resnet50','fasterrcnn_mobilenetv3_large_320_fpn',
# 'fasterrcnn_mobilenetv3_large_fpn','fasterrcnn_squeezenet1_0','fasterrcnn_squeezenet1_1'

MODEL_NAME = 'fasterrcnn_resnet18'
