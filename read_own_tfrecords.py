import os
import re
import numpy as np 
import tensorflow as tf 
from PIL import Image 


def read_tfrecords_and_decode(filename, batch_size):
	"""
	Args:
		filename: tfrecords file, string
		batch_size: int
	Return:
		dataset iterator , (img_tensor, label_tensor)
	"""
	def parser(record):
		features =tf.io.parse_single_example(record, features={
			'label': tf.io.FixedLenFeature([], tf.int64),
			'img_raw':tf.io.FixedLenFeature([], tf.string)
			})
		image = tf.decode_raw(features['img_raw'], tf.uint8)
		img = tf.reshape(image, [256, 256, 3])
		img = tf.cast(img, tf.float32)
		label = tf.cast(features['label'], tf.int32)

		return img, label
	# filename_queue = tf.train.string_input_producer([filename], shuffle=True)
	# reader = tf.TFRecordReader()
	dataset = tf.data.TFRecordDataset(filename)
	dataset = dataset.map(parser).repeat().batch(batch_size).shuffle(buffer_size=3)

	# iterator = dataset.make_one_shot_iterator()
	# img_input, label = iterator.get_next()
	# return img_input, label

	iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
	dataset_next = iterator.get_next()
	return dataset_next


if __name__ == '__main__':
	read_tfrecords_and_decode('test.tfrecords', 4)