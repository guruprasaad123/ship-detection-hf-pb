import torch
import cv2
import numpy as np
import pandas as pd
import os
import glob as glob
from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, TEST_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform, get_submission_transform
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x,y,w,h]

# the dataset class
class ShipsDataset(Dataset):
	
	def __init__(self, dir_path, label, width, height, classes, submission=False, transforms=None):
		self.transforms = transforms
		self.dir_path = dir_path
		self.height = height
		self.width = width
		self.classes = classes
		self.submission = submission

		if label == None:
			
			# get all the image paths in sorted order
			self.image_paths = glob.glob(f"{self.dir_path}/*.png")
			self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
			self.all_images = sorted(self.all_images)
			self.label_data = pd.DataFrame(self.all_images,columns=['id'])
			self.label_data['group_idx'] = self.label_data.groupby(['id']).ngroup()
		
		else:
			self.label_data = pd.read_csv(label)

			self.label_data['group_idx'] = self.label_data.groupby(['id']).ngroup()
		
	
	def __getitem__(self, idx):
		# capture the image name and the full image path

		image_annotations = self.label_data[ self.label_data['group_idx'] == idx ]

		image_name = image_annotations['id'].unique().tolist()[0]

		image_name_id = int( image_name.replace('.png','') )

		image_path = os.path.join(self.dir_path, image_name)
		# # read the image
		image = cv2.imread(image_path)
		# convert BGR to RGB color format
		image_resized = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
		# image_resized = cv2.resize(image_resized, (self.width, self.height))
		image_resized /= 255.0

		# image_resized = image_resized.transpose((2,0,1))

		# print('image_resized',image_resized.shape)

		# # capture the corresponding XML file for getting the annotations
		# annot_filename = image_name[:-4] + '.xml'
		# annot_file_path = os.path.join(self.dir_path, annot_filename)
		
		boxes = []
		labels = []

		# get the height and width of the image
		image_width = image.shape[1]
		image_height = image.shape[0]

		if self.submission == False:

				for annnotation in image_annotations.to_dict('records'):

					label = self.classes.index( 'ship' )

					labels.append( label )

					# xmin = left corner x-coordinates
					xmin = int( annnotation['xmin'] )
					# xmax = right corner x-coordinates
					xmax = int( annnotation['xmax'] )
					# ymin = left corner y-coordinates
					ymin = int( annnotation['ymin'] )
					# ymax = right corner y-coordinates
					ymax = int( annnotation['ymax'] )

					xmin_final = ( (xmin/image_width)*self.width )
					xmax_final = ( (xmax/image_width)*self.width )
					ymin_final = ( (ymin/image_height)*self.height )
					ymax_final = ( (ymax/image_height)*self.height )

					boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

				# # bounding box to tensor
				boxes = torch.as_tensor(boxes, dtype=torch.float32)
				# area of the bounding boxes
				area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
				# no crowd instances
				iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
				# labels to tensor
				labels = torch.as_tensor(labels, dtype=torch.int64)
				# prepare the final `target` dictionary
				target = {}
				target["boxes"] = boxes
				target["labels"] = labels
				target["area"] = area
				target["iscrowd"] = iscrowd
				image_id = torch.tensor([idx])
				target["image_id"] = image_id
				target["image_name_id"] = torch.tensor([image_name_id])

				# apply the image transforms
				if self.transforms:
					sample = self.transforms(image = image_resized,
											 bboxes = target['boxes'],
											 labels = labels)
					image_resized = sample['image']
					target['boxes'] = torch.Tensor(sample['bboxes'])

				return image_resized, target

		else:
			target = {}
			target["image_name_id"] = torch.tensor([image_name_id])

			# apply the image transforms
			if self.transforms:
				sample = self.transforms(image = image_resized)
				image_resized = sample['image']

			return image_resized, target

	def __len__(self):
		return self.label_data['group_idx'].unique().shape[0]

def create_train_val_split(train_path = os.path.join('..','input','.extras','train.csv'), output_path = os.path.join('..','input','.extras')):
	"""
	To create training , validation , testing split using folds method
	"""

	train_data = pd.read_csv(train_path)

	req_cols = train_data.columns.tolist()

	print('init',len(train_data.groupby('id')))

	gkfold = GroupKFold(n_splits=8)
	train_data['fold'] = -1

	for fold, (_, valid_idx) in enumerate(
		gkfold.split(train_data, groups=train_data.id)
	):
		train_data.loc[valid_idx, 'fold'] = fold
		
	train_df = train_data.query('fold <=5')

	test_df = train_data.query('fold == 6')

	valid_df = train_data.query('fold == 7')

	print( len(train_df.groupby('id')), len(test_df.groupby('id')), len(valid_df.groupby('id')) )

	print( len(train_df.groupby('id')) + len(test_df.groupby('id')) + len(valid_df.groupby('id')) )

	train_df[req_cols].to_csv( os.path.join( output_path ,'train_fold.csv') , index=False )

	test_df[req_cols].to_csv( os.path.join( output_path ,'test_fold.csv') , index=False )

	valid_df[req_cols].to_csv( os.path.join( output_path ,'val_fold.csv') , index=False )

	return train_df , test_df , valid_df

# prepare the final datasets and data loaders

def create_train_dataset():
	"""
	create train dataset with transformations
	"""

	train_dataset = ShipsDataset(
		TRAIN_DIR, os.path.join('..','input','.extras','train_fold.csv'),
		RESIZE_TO, RESIZE_TO, CLASSES, False, get_train_transform()
		)

	return train_dataset

def create_valid_dataset(labels_file_name):
	"""
	create test dataset with the labels_file_name and validation transformation
	"""

	test_dataset = ShipsDataset(
		TRAIN_DIR, os.path.join('..','input','.extras',labels_file_name),
		RESIZE_TO, RESIZE_TO, CLASSES, False, get_valid_transform()
		)

	return test_dataset

def create_submission_dataset():
	"""
	crete submission dataset with
	"""

	submit_dataset = ShipsDataset(
		TEST_DIR, None,
		RESIZE_TO, RESIZE_TO, CLASSES, submission=True, transforms=get_submission_transform()
		)

	return submit_dataset


def create_train_loader(train_dataset, num_workers=0):
	train_loader = DataLoader(
		train_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=collate_fn
	)

	return train_loader


def create_valid_loader(valid_dataset, num_workers=0):
	valid_loader = DataLoader(
		valid_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=collate_fn
	)

	return valid_loader

def create_submission_loader(submit_dataset, num_workers=0):

	submit_loader = torch.utils.data.DataLoader(
		submit_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=num_workers,
		collate_fn=collate_fn
		)

	return submit_loader

if __name__ == '__main__':

	# prepare the final datasets and data loaders
	
	train_dataset = ShipsDataset(TRAIN_DIR, os.path.join('..','input','.extras','train.csv'), RESIZE_TO, RESIZE_TO, CLASSES,)
	valid_dataset = ShipsDataset(TEST_DIR, None, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
	
	train_loader = DataLoader(
		train_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=0,
		collate_fn=collate_fn
	)
	
	valid_loader = DataLoader(
		valid_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=0,
		collate_fn=collate_fn
	)

	print(f"Number of training samples: {len(train_dataset)}")
	print(f"Number of validation samples: {len(valid_dataset)}\n")

	print(f"{train_dataset[0]}")

	# function to visualize a single sample
	def visualize_sample(image, target, filename):
		
		boxs = target['boxes']
		labels = target['labels']

		for x,item in enumerate(zip(boxs,labels)):

			box, label = item

			# print('label',label,box)

			label = CLASSES[label]

			cv2.rectangle(
				image, 
				(int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
				(0, 255, 0), 1
			)
			cv2.putText(
				image, label, (int(box[0]), int(box[1]-5)), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
			)

		cv2.imwrite(filename, 255*image)

		# cv2.imshow('Image', image)
		# cv2.waitKey(0)
	
	# create_train_val_split()

	NUM_SAMPLES_TO_VISUALIZE = 10
	for i in range(NUM_SAMPLES_TO_VISUALIZE):
		image, target = train_dataset[i]
		filename = os.path.join('..','visualization', f'image_{i}.jpg')
		visualize_sample(image, target, filename)


