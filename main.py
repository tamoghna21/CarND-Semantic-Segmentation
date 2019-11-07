#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from math import ceil
import sys
from tqdm import tqdm

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
	warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
	print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
	
print("Press Enter to continue")
input()

def load_vgg(sess, vgg_path):
	"""
	Load Pretrained VGG Model into TensorFlow.
	:param sess: TensorFlow Session
	:param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
	:return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
	"""
	# TODO: Implement function
	#   Use tf.saved_model.loader.load to load the model and weights
	vgg_tag = 'vgg16'
	tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
	
	vgg_input_tensor_name = 'image_input:0'
	vgg_keep_prob_tensor_name = 'keep_prob:0'
	vgg_layer3_out_tensor_name = 'layer3_out:0'
	vgg_layer4_out_tensor_name = 'layer4_out:0'
	vgg_layer7_out_tensor_name = 'layer7_out:0'
	
	graph = tf.get_default_graph()
	#graph = tf.Graph()
	
	input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
	keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
	vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
	vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
	vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
	
	return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
	
print("End of load_vgg....Press Enter to continue")
input()


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
	"""
	Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
	:param vgg_layer3_out: TF Tensor for VGG Layer 3 output
	:param vgg_layer4_out: TF Tensor for VGG Layer 4 output
	:param vgg_layer7_out: TF Tensor for VGG Layer 7 output
	:param num_classes: Number of classes to classify
	:return: The Tensor for the last layer of output
	"""
	# TODO: Implement function
	
	# Regularizers and initializers
	initializer = lambda: tf.truncated_normal_initializer(stddev=0.01)
	regularizer = lambda: tf.contrib.layers.l2_regularizer(1e-5)
	
	# 1x1 convolution
	layer7_out = tf.layers.conv2d(
	inputs=vgg_layer7_out,
	filters=num_classes,
	kernel_size=1,
	padding='same',
	kernel_regularizer=regularizer(),
	kernel_initializer=initializer(),
	name = 'layer7_out')
	
	
	# Upsample
	layer7_up = tf.layers.conv2d_transpose(
	inputs=layer7_out,
	filters=num_classes,
	kernel_size=4,
	strides=(2, 2),
	padding='same',
	kernel_regularizer=regularizer(),
	kernel_initializer=initializer(),
	name = 'layer7_up')
	
	# Scaling of pooling layer 4
	vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.0001, name='vgg_layer4_out_scaled')
	# 1x1 convolution
	layer4_out = tf.layers.conv2d(
    inputs=vgg_layer4_out_scaled,
    filters=num_classes,
    kernel_size=1,
    padding='same',
    kernel_regularizer=regularizer(),
    kernel_initializer=initializer(),
    name = 'layer4_out')

	# Skip layer
	skip_1 = tf.add(layer7_up, layer4_out)

	# Upsample
	skip_1_up = tf.layers.conv2d_transpose(
	inputs=skip_1,
	filters=num_classes,
	kernel_size=4,
	strides=(2, 2),
	padding='same',
	kernel_regularizer=regularizer(),
	kernel_initializer=initializer(),
	name = 'skip_1_up')

	# Scaling of pooling layer 3
	vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='vgg_layer3_out_scaled')
	# 1x1 convolution
	layer3_out = tf.layers.conv2d(
    inputs=vgg_layer3_out_scaled,
    filters=num_classes,
    kernel_size=1,
    padding='same',
    kernel_regularizer=regularizer(),
    kernel_initializer=initializer(),
    name='layer3_out')

	# Skip layer
	skip_2 = tf.add(skip_1_up, layer3_out)

	# Upsampled final
	nn_last_layer = tf.layers.conv2d_transpose(
	inputs=skip_2,
	filters=num_classes,
	kernel_size=16,
	strides=(8, 8),
	padding='same',
	kernel_regularizer=regularizer(),
	kernel_initializer=initializer(),
	name='nn_last_layer')
	
	return nn_last_layer
	

print("End of layers....Press Enter to continue")
input()


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
	"""
	Build the TensorFLow loss and optimizer operations.
	:param nn_last_layer: TF Tensor of the last layer in the neural network
	:param correct_label: TF Placeholder for the correct label image
	:param learning_rate: TF Placeholder for the learning rate
	:param num_classes: Number of classes to classify
	:return: Tuple of (logits, train_op, cross_entropy_loss)
	"""
	# TODO: Implement function
	logits = tf.reshape(nn_last_layer, (-1, num_classes),name='logits')
	truth = tf.reshape(correct_label, (-1, num_classes))
	predicts = tf.nn.softmax(logits, name='predicts')
	#global_step = tf.Variable(0, name='global_step', trainable=False)
	
	# Cross-entropy operation
	#cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=truth)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=truth)
	cross_entropy_loss = tf.reduce_mean(cross_entropy)
	tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
	
	# Regularization loss
	l2_loss = tf.losses.get_regularization_loss()
	tf.summary.scalar('l2_loss', l2_loss)
	
	total_loss = cross_entropy_loss + l2_loss
	tf.summary.scalar('total_loss', total_loss)
	
	#train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, global_step)
	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
	
	# Merge summary operation
	merged = tf.summary.merge_all()
	return logits, train_op, cross_entropy_loss

print("End of optimize....Press Enter to continue")
input()

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
			 correct_label, keep_prob, learning_rate,saver):
	"""
	Train neural network and print out the loss during training.
	:param sess: TF Session
	:param epochs: Number of epochs
	:param batch_size: Batch size
	:param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
	:param train_op: TF Operation to train the neural network
	:param cross_entropy_loss: TF Tensor for the amount of loss
	:param input_image: TF Placeholder for input images
	:param correct_label: TF Placeholder for label images
	:param keep_prob: TF Placeholder for dropout keep probability
	:param learning_rate: TF Placeholder for learning rate
	"""
	# TODO: Implement function
	min_epochs = 20
	best_loss = 1e9
	failure = 0
	patience = 4
	
	sess.run(tf.global_variables_initializer())
	
	for e in range(epochs):
		epoch_loss = 0
		num_images = 0
		sys.stdout.flush()
		for images, labels in get_batches_fn(batch_size):
			_, loss = sess.run([
			  train_op,
			  cross_entropy_loss], feed_dict={
				  input_image: images,
				  correct_label: labels,
				  keep_prob: 0.5,
				  learning_rate: 1e-4})
			#writer.add_summary(summary, step)
			epoch_loss += loss * len(images)
			num_images += len(images)
			
		epoch_loss /= num_images
		sys.stderr.flush()
		print('Epoch {} loss: {:.3f}'.format(e + 1, epoch_loss))
		if e >= min_epochs and epoch_loss > best_loss:
			if failure == patience:
				break
			failure += 1
		else:
			failure = 0
			best_loss = epoch_loss
			print('Saving model')
			saver.save(sess, './model.ckpt')
	#pass

print("End of train_nn...Press Enter to continue")
input()

def run():
	num_classes = 2
	image_shape = (160, 576)
	data_dir = './data'
	runs_dir = './runs'
	#tests.test_for_kitti_dataset(data_dir)# This will also download the kitti dataset, if not already there
	
	correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
	learning_rate = tf.placeholder(tf.float32)
	#keep_prob = tf.placeholder(tf.float32)
	
	epochs = 50
	batch_size = 4

	# Download pretrained vgg model
	#helper.maybe_download_pretrained_vgg(data_dir)

	# OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
	# You'll need a GPU with at least 10 teraFLOPS to train on.
	#  https://www.cityscapes-dataset.com/

	with tf.Session() as sess:
		# Path to vgg model
		vgg_path = os.path.join(data_dir, 'vgg')
		# Create function to get batches
		get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape,augment=False)

		# OPTIONAL: Augment Images for better results
		#  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

		# TODO: Build NN using load_vgg, layers, and optimize function
		input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
		
		nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
		
		logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
		
		saver = tf.train.Saver()



		# TODO: Train NN using the train_nn function
		train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
			 correct_label, keep_prob, learning_rate,saver)

		# TODO: Save inference data using helper.save_inference_samples
		#helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)



##Finally training is initiated
if __name__ == '__main__':
	run()
print("End")
