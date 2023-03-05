import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from torch_utils import utils
from torch_utils.coco_eval import CocoEvaluator
from torch_utils.coco_utils import get_coco_api_from_dataset
from tqdm import tqdm
import pandas as pd
import sys


def train_one_epoch(
	model, 
	optimizer, 
	data_loader, 
	device, 
	epoch, 
	train_loss_hist,
	print_freq, 
	scaler=None,
	scheduler=None,
	writer=None,
):
	model.train()
	metric_logger = utils.MetricLogger(delimiter="  ")
	metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
	header = f"Epoch: [{epoch}]"

	# List to store batch losses.
	batch_loss_list = []

	lr_scheduler = None
	if epoch == 0:
		warmup_factor = 1.0 / 1000
		warmup_iters = min(1000, len(data_loader) - 1)

		lr_scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer, start_factor=warmup_factor, total_iters=warmup_iters
		)

	step_counter = 0
	for idx,(images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
		with torch.cuda.amp.autocast(enabled=scaler is not None):
			loss_dict = model(images, targets)
			losses = sum(loss for loss in loss_dict.values())

		# reduce losses over all GPUs for logging purposes
		loss_dict_reduced = utils.reduce_dict(loss_dict)
		losses_reduced = sum(loss for loss in loss_dict_reduced.values())

		loss_value = losses_reduced.item()

		if not math.isfinite(loss_value):
			print(f"Loss is {loss_value}, stopping training")
			print(loss_dict_reduced)
			sys.exit(1)

		optimizer.zero_grad()
		if scaler is not None:
			scaler.scale(losses).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			losses.backward()
			optimizer.step()

		if lr_scheduler is not None:
			lr_scheduler.step()

		metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
		metric_logger.update(lr=optimizer.param_groups[0]["lr"])

		batch_loss_list.append(loss_value)
		train_loss_hist.send(loss_value)

		if scheduler is not None:
			scheduler.step(epoch + (step_counter/len(data_loader)))

		if writer is not None:
		
			global_step = epoch * len(data_loader) + idx

			running_loss = sum(batch_loss_list)/len(batch_loss_list)

			writer.add_scalar('training_loss',running_loss,global_step)

	return metric_logger, batch_loss_list


def _get_iou_types(model):
	model_without_ddp = model
	if isinstance(model, torch.nn.parallel.DistributedDataParallel):
		model_without_ddp = model.module
	iou_types = ["bbox"]
	if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
		iou_types.append("segm")
	if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
		iou_types.append("keypoints")
	return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
	n_threads = torch.get_num_threads()
	# FIXME remove this and make paste_masks_in_image run on the GPU
	torch.set_num_threads(1)
	cpu_device = torch.device("cpu")
	model.eval()
	metric_logger = utils.MetricLogger(delimiter="  ")
	header = "Test:"

	coco = get_coco_api_from_dataset(data_loader.dataset)
	iou_types = _get_iou_types(model)
	coco_evaluator = CocoEvaluator(coco, iou_types)

	for images, targets in metric_logger.log_every(data_loader, 100, header):
		images = list(img.to(device) for img in images)

		if torch.cuda.is_available():
			torch.cuda.synchronize()
		model_time = time.time()
		outputs = model(images)

		outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
		model_time = time.time() - model_time

		res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
		evaluator_time = time.time()
		coco_evaluator.update(res)
		evaluator_time = time.time() - evaluator_time
		metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)
	coco_evaluator.synchronize_between_processes()

	# accumulate predictions from all images
	coco_evaluator.accumulate()
	coco_evaluator.summarize()
	torch.set_num_threads(n_threads)
	return coco_evaluator

@torch.no_grad()
def submit_results(model, data_loader, device='cpu', output_path=None, CLASSES=list([])):

	submission_list = list([])

	model.eval()

	model.to(device)

	for images,targets in tqdm(data_loader):

		images = list(img.to(device) for img in images)

		target_list_batch = [ target['image_name_id'].item() for target in targets ]

		# print('target_list_batch',target_list_batch)

		torch.cuda.synchronize()
		model_time = time.time()
		outputs = model(images)

		outputs = [ {k: v.cpu().detach().numpy().tolist() for k, v in t.items()} for t in outputs ]

		# print('outputs',outputs)

		model_time = time.time() - model_time

		for i,(t,o) in enumerate(zip(target_list_batch,outputs)):

			obj = pd.DataFrame( o['boxes'] , columns=['XMin','XMax','YMin','YMax'] )
			obj['LabelName'] = o['labels']
			obj['Conf'] = o['scores']
			obj['ImageID'] = t

			if obj.shape[0] > 0:

				submission_list.append(obj)

		# sys.exit(1)

	cols = ['ImageID','LabelName','Conf','XMin','XMax','YMin','YMax']

	submission_df = pd.concat(submission_list)

	submission_df = submission_df[cols]

	submission_df['ImageID'] = submission_df['ImageID'].apply( lambda x : f'{x}.jpg' )

	submission_df['LabelName'] = submission_df['LabelName'].apply( lambda x : CLASSES[x] )

	if output_path:

		submission_df.to_csv(output_path, index=False )

	return submission_df
