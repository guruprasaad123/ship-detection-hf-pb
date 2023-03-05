from torch_utils.engine import (
	train_one_epoch, evaluate, submit_results
)
from config import (
	DEVICE, NUM_CLASSES,
	BATCH_SIZE,
	NUM_EPOCHS, NUM_WORKERS,
	LOG_DIR,
	OUT_DIR, VISUALIZE_TRANSFORMED_IMAGES,
	CLASSES, SAVE_FOR_EVERY_EPOCH,
	MODEL_NAME
)
from datasets import (
	create_train_dataset, create_valid_dataset, 
	create_train_loader, create_valid_loader,
	create_submission_dataset, create_submission_loader
)
from models.fasterrcnn_squeezenet1_1 import create_model
from custom_utils import (
	save_model, 
	save_train_loss_plot,
	Averager, show_tranformed_image, collate_fn, get_model_by_name
)
from logger import get_logger

import torch
import time
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

	last_epoch = opt_state = model_state = last_epoch = None

	LEARNING_RATE = 0.0001
	WEIGHT_DECAY = 0.0005

	if os.path.exists(LOG_DIR) == False:
		os.makedirs(LOG_DIR)

	logger = get_logger(__name__, log_path=os.path.join(LOG_DIR, f'{MODEL_NAME}_{time.time()}.log'), console=True)

	CHECKPOINT_PATH = os.path.join( '..','checkpoint' , MODEL_NAME )
	
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

	# create datasets
	train_dataset = create_train_dataset()
	test_dataset = create_valid_dataset('test_fold.csv')
	valid_dataset = create_valid_dataset('val_fold.csv')
	submission_dataset = create_submission_dataset()

	# create dataloaders
	train_loader = create_train_loader(train_dataset, NUM_WORKERS)
	test_loader = create_train_loader(test_dataset, NUM_WORKERS)
	valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
	submission_loader = create_submission_loader(submission_dataset, NUM_WORKERS)

	print(f"Number of training samples: {len(train_dataset)}")
	print(f"Number of testing samples: {len(test_dataset)}\n")
	print(f"Number of validation samples: {len(valid_dataset)}\n")

	print(f"Number of submission samples: {len(submission_dataset)}\n")

	if VISUALIZE_TRANSFORMED_IMAGES:
		show_tranformed_image(train_loader)

	# Initialize the Averager class.
	train_loss_hist = Averager()
	# Train and validation loss lists to store loss values of all
	# iterations till ena and plot graphs for all iterations.
	train_loss_list = []

	# Initialize the model and move to the computation device.

	custom_model = get_model_by_name(MODEL_NAME)

	model = custom_model(num_classes=NUM_CLASSES)
	model = model.to(DEVICE)

	if model_state:
		model.load_state_dict(model_state)

	log_dir = os.path.join( "..", "run", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") )
	
	comment = ' model = {} batch_size = {} lr = {} weight_decay = {} optimizer = {}'.format(
														MODEL_NAME,
														 BATCH_SIZE,
														 LEARNING_RATE,
														 WEIGHT_DECAY,
														 'AdamW'
														 )

	writer = SummaryWriter(comment=comment,log_dir=log_dir)

	# Total parameters and trainable parameters.
	total_params = sum(p.numel() for p in model.parameters())

	print(f"{total_params:,} total parameters.")
	
	total_trainable_params = sum(
		p.numel() for p in model.parameters() if p.requires_grad)

	print(f"{total_trainable_params:,} training parameters.\n")
	
	# Get the model parameters.
	params = [p for p in model.parameters() if p.requires_grad]
	
	# Define the optimizer.
	
	# optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
	optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

	if opt_state:
		optimizer.load_state_dict(opt_state)

	logger.info('Starting with {} instances'.format(BATCH_SIZE))

	# LR will be zero as we approach `steps` number of epochs each time.
	# If `steps = 5`, LR will slowly reduce to zero every 5 epochs.
	steps = NUM_EPOCHS + 25
	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
		optimizer, 
		T_0=steps,
		T_mult=1,
		verbose=True
	)

	for epoch in range( last_epoch+1 , NUM_EPOCHS + 1 ) if last_epoch else range(1, NUM_EPOCHS + 1):
		train_loss_hist.reset()

		_, batch_loss_list = train_one_epoch(
			model, 
			optimizer, 
			train_loader, 
			DEVICE, 
			epoch, 
			train_loss_hist,
			print_freq=20,
			scheduler=scheduler,
			writer=writer,
		)

		logger.info('Mean Training Loss on Epoch:{} = {}'.format(epoch,sum(batch_loss_list)/len(batch_loss_list)))

		evaluate(model, test_loader, device=DEVICE)

		evaluate(model, valid_loader, device=DEVICE)

		# save the model for SAVE_FOR_EVERY_EPOCH
		if epoch % SAVE_FOR_EVERY_EPOCH == 0:

			torch.save( {
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			# 'loss': loss,
			} , os.path.join( CHECKPOINT_PATH , '{}-{:04d}.bin'.format(MODEL_NAME,epoch) ) )


		# Add the current epoch's batch-wise lossed to the `train_loss_list`.
		train_loss_list.extend(batch_loss_list)

		# Save the current epoch model.
		save_model(OUT_DIR, epoch, model, optimizer)

		# Save loss plot.
		save_train_loss_plot(OUT_DIR, train_loss_list)

	torch.save( {
	'epoch': epoch,
	'model_state_dict': model.state_dict(),
	'optimizer_state_dict': optimizer.state_dict(),
	# 'loss': loss,
	} , os.path.join( CHECKPOINT_PATH , '{}-{:04d}.bin'.format(MODEL_NAME,epoch) ) )

	output_path = os.path.join('..', 'input', 'test', f'submission_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

	submit_results(model, submission_loader, device=DEVICE, output_path=output_path, CLASSES=CLASSES)
