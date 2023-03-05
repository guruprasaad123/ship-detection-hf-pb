from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from config import CLASSES, RESIZE_TO, TRAIN_DIR, BATCH_SIZE, TEST_DIR

from model import create_model
from utils import Averager, collate_fn, get_train_transform, get_valid_transform, get_submission_transform
from tqdm import tqdm
# from datasets import train_loader, valid_loader
from datasets import PotholesDataset
import torch
import matplotlib.pyplot as plt
import time
from model import create_model
import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from logger import get_logger
from torch import nn , optim
import sys

plt.style.use('ggplot')

from train_utils import train_one_epoch
from test_utils import evaluate_one_epoch

# function for running training iterations
def train(train_data_loader, model):
	print('Training')
	global train_itr
	global train_loss_list
	
	 # initialize tqdm progress bar
	prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
	
	for i, data in enumerate(prog_bar):
		optimizer.zero_grad()
		images, targets = data
		
		images = list(image.to(DEVICE) for image in images)
		targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
		loss_dict = model(images, targets)
		losses = sum(loss for loss in loss_dict.values())
		loss_value = losses.item()
		train_loss_list.append(loss_value)
		train_loss_hist.send(loss_value)
		losses.backward()
		optimizer.step()
		train_itr += 1
	
		# update the loss value beside the progress bar for each iteration
		prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
	return train_loss_list

def train_single_epoch(model,scaler,optimizer,train_loader,writer,logger,device='cpu',epoch=0):
	"""
	This utility is for training the model for a single epoch
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
		printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
	"""
	loop = tqdm(train_loader,leave=True)
	losses = []

	train_loss_list , train_iteration = 0, 0

	running_loss = 0
	start_point = time.time()
	at_start = True

	for idx,(inputs,targets) in enumerate(loop):

		with torch.cuda.amp.autocast():

			optimizer.zero_grad()

			# inputs , targets = Variable(inputs).to(device) , Variable(targets).to(device)

			images = list( Variable(inputs).to(device) for image in images)
			targets = [{k: Variable(v).to(device) for k, v in t.items()} for t in targets]

			loss_dict = model(images, targets)

		if at_start == True:
			# create grid of images

			sample_inputs = inputs

			img_grid = make_grid(sample_inputs)

			# write to tensorboard
			# writer.add_image('sample Images', img_grid)

			# writer.add_graph(model, sample_inputs)

			model.to(device)

			at_start = False

		losses = sum(loss for loss in loss_dict.values())
		loss = losses.item()

		train_loss_list.append(loss)
		train_loss_hist.send(loss)

		scaler.scale(losses).backward()

		# loss.backward()

		# optimizer.step()

		scaler.step(optimizer)
		scaler.update()

		loop.set_postfix(loss=loss.item())

		running_loss = sum(train_loss_list)/len(train_loss_list)

		train_loss += loss.item()
 
		global_step = epoch * len(train_loader) + idx

		train_iteration += 1

		writer.add_scalar('training_loss',running_loss,global_step)

	# train_accuracy = (100 * train_correct)/train_total

	train_loss = train_loss / len(train_loader)

	mean_loss = sum(train_loss_list)/len(train_loss_list)

	logger.info('Training Mean Loss on Epoch:{} = {}'.format(epoch,mean_loss))

	logger.info('train-loss : %.5f' % (train_accuracy,train_loss))

	logger.info('elapsed time: %ds' % (time.time() - start_point))

	print('[*] Training Mean Loss : {}'.format(mean_loss))

	return train_loss_list


@torch.no_grad()
def submit_results(model, data_loader, device):

	model.eval()

	model.to(device)

	for images,targets in data_loader:

		images = list(img.to(device) for img in images)

		print('images',images)

		torch.cuda.synchronize()
		model_time = time.time()
		outputs = model(images)

		print('outputs',outputs)

		outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

		model_time = time.time() - model_time

		print('image_name_id',targets)

		sys.exit(1)


if __name__ == '__main__':

	last_epoch = opt_state = model_state = last_epoch = None

	CHECKPOINT_PATH = os.path.join( 'checkpoint' , 'fast-rcnn' )
	
	# if the checkpoint path does not exists , then create it
	if not os.path.exists( CHECKPOINT_PATH ):
		os.makedirs( CHECKPOINT_PATH )
	# if checkpoint path already exists
	else:
		dirs = os.listdir( CHECKPOINT_PATH )

		if len(dirs) > 0:

			latest_checkpoint = max(dirs)

			checkpoint = torch.load( os.path.join(CHECKPOINT_PATH,latest_checkpoint) )

			model_state = checkpoint['model_state_dict']
			opt_state = checkpoint['optimizer_state_dict']
			last_epoch = checkpoint['epoch']
			# loss = checkpoint['loss']

			print(' [*] Model Restored from {} Epoch \n'.format(last_epoch) )

			model_restored = True

	# train on the GPU or on the CPU, if a GPU is not available
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	model = create_model( len(CLASSES) )

	if model_state:
		model.load_state_dict(model_state)

	train_dataset = PotholesDataset(TRAIN_DIR, os.path.join('train','train_fold.csv'), RESIZE_TO, RESIZE_TO, CLASSES, False, get_train_transform())

	train_data_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=2,
		shuffle=True,
		num_workers=4,
		collate_fn=collate_fn
		) 

	test_dataset = PotholesDataset(TRAIN_DIR, os.path.join('train','test_fold.csv'), RESIZE_TO, RESIZE_TO, CLASSES, False, get_valid_transform())

	test_data_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=2,
		shuffle=True,
		num_workers=4,
		collate_fn=collate_fn
		) 

	val_dataset = PotholesDataset(TRAIN_DIR, os.path.join('train','val_fold.csv'), RESIZE_TO, RESIZE_TO, CLASSES, False,get_valid_transform())

	val_data_loader = torch.utils.data.DataLoader(
		val_dataset,
		batch_size=2,
		shuffle=True,
		num_workers=4,
		collate_fn=collate_fn
		)

	submit_dataset = PotholesDataset(TEST_DIR, None, RESIZE_TO, RESIZE_TO, CLASSES, submission=True, transforms=get_submission_transform())

	submit_data_loader = torch.utils.data.DataLoader(
		submit_dataset,
		batch_size=2,
		shuffle=True,
		num_workers=4,
		collate_fn=collate_fn
		) 

	# move model to the right device
	model.to(device)

	# construct an optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005,
								momentum=0.9, weight_decay=0.0005)

	if opt_state:
		optimizer.load_state_dict(opt_state)

	# and a learning rate scheduler
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
												   step_size=3,
												   gamma=0.1)

	# let's train it for 10 epochs
	num_epochs = 11

	for epoch in range( last_epoch+1 , num_epochs + 1 ) if last_epoch else range(1, num_epochs + 1):
		# train for one epoch, printing every 10 iterations
		train_results = train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)

		print('train_results',train_results)

		# update the learning rate
		lr_scheduler.step()
		
		# evaluate on the test dataset
		test_results = evaluate_one_epoch(model, test_data_loader, device=device)

		print('test_results',test_results)
		
		# evaluate on the validation dataset
		val_results = evaluate_one_epoch(model, val_data_loader, device=device)

		print('val_results',val_results)

		# save the model
		torch.save( {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		# 'loss': loss,
		} , os.path.join( CHECKPOINT_PATH , 'fast-rcnn-{:04d}.bin'.format(epoch) ) )

	submit_results(model, submit_data_loader, device='cpu')
