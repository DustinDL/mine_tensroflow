import os
import re
import numpy as np 
import tensorflow as tf 
from PIL import Image 
from tqdm import tqdm


def get_file(data_dir):
	"""
	Args:
		data_dir:
			/class1
				/picture.....
			/class2
				/picture.....
			......
	Returns:
		[image_path],[label] string
	"""
	label_plant = []
	image_plant = []
	is_root = 1
	for root, dirs, files in os.walk(data_dir):
		if is_root == 1:
			is_root = 0
		else:
			for file in files:
				image_plant.append(str(root + '/' + file))
				label_plant.extend(re.findall(r'\/(.+)', root))
	tmp = np.array([image_plant, label_plant])
	tmp = tmp.transpose() 
	np.random.shuffle(tmp) 

	image_list = list(tmp[:, 0])
	label_list = list(tmp[:, 1])
	return image_list, label_list


def make_tfrecords(data_dir, image_size, output):
	"""
	Args:
		data_dir:
			/class1
				/picture.....
			/class2
				/picture.....
			......
		image_size: int, 128, 256
		output: string
	"""
	writer = tf.io.TFRecordWriter(output)
	label_str = os.listdir(data_dir)
	for index, name in tqdm(enumerate(label_str)):
		label_path = data_dir + '/' + name
		for img_name in os.listdir(label_path):
			img_path = label_path + '/' + img_name
			img = Image.open(img_path)
			img = img.resize((image_size, image_size))
			# print(np.shape(img))
			img_raw = img.tobytes()
			example = tf.train.Example(features=tf.train.Features(feature={
				"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
				"img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
				}))
			writer.write(example.SerializeToString())

	writer.close()


if __name__ == '__main__':
	# image_list, label_list = get_file('test')
	make_tfrecords('test', 256, 'test.tfrecords')