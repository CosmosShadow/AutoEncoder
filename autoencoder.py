# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
from scipy import misc
import random
import data_interface


# 超参
crop_size = 8
count = 10000
hidden_size = 7*7
spare_rate = 0.1
decay_lambda = 0.0001  # weight decay parameter
sparse_beta = 3  # weight of sparsity penalty term
lr = 0.01
batch_size = 128

# 准备数据
img = misc.imread('1.png', flatten=True)
print img.shape
# misc.imsave('2.png', img)
img = img / 255.0
data = np.zeros((count, crop_size, crop_size))
for i in range(count):
	y = random.randint(0, img.shape[0]-crop_size-1)
	x = random.randint(0, img.shape[1]-crop_size-1)
	data[i] = img[y:y+crop_size, x:x+crop_size]

# 数据放在标准接口中
dataSets = data_interface.DataSets()
dataSets.train.set_x(data)
dataSets.train.set_y(data)

x = tf.placeholder(tf.float32, [None, crop_size, crop_size])
y = tf.placeholder(tf.float32, [None, crop_size, crop_size])

x_reshape = tf.reshape(x, [-1, crop_size*crop_size])
y_reshape = tf.reshape(y, [-1, crop_size*crop_size])

x_wrap = pt.wrap(x_reshape)
with pt.defaults_scope(activation_fn=tf.nn.sigmoid, l2loss=decay_lambda):
	hidden = x_wrap.fully_connected(hidden_size)
	output = hidden.fully_connected(crop_size*crop_size)

# 所有变量
for v in tf.all_variables():
	print v.name
weights = [v for v in tf.all_variables() if v.name == "fully_connected/weights:0"][0]

# 添加稀疏性
active_mean = tf.reduce_mean(hidden, -1)
spare_loss = spare_rate * tf.log(spare_rate/active_mean) + (1-spare_rate) * tf.log((1-spare_rate)/(1-active_mean))
spare_loss_scale = tf.reduce_mean(spare_loss)

loss_regression = tf.reduce_mean(tf.square(tf.sub(output, y_reshape)))

loss = loss_regression + spare_loss_scale * sparse_beta

train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(loss)

# 最大激活权重
def max_activate_weight(weights_):
	image = np.copy(weights_)
	for i in range(image.shape[0]):
		base = np.sqrt(np.square(image[i]).sum())
		image[i] = np.absolute(image[i]) / base
		# image[i] -= image[i].min()
		# image[i] /= image[i].max()
	return image

def images2one(data, padsize=1, padval=1.0):
	# data -= data.min()
	# data /= data.max()
	n = int(np.ceil(np.sqrt(data.shape[0])))
	h = n * data.shape[1] + (n - 1)
	w = n * data.shape[2] + (n - 1)
	unit_shape = [h, w] + list(data.shape)[3:]
	unit = np.ones(unit_shape, dtype=data.dtype) * padval
	for i in range(data.shape[0]):
		h_i = (i / n) * (data.shape[1] +1)
		w_i = (i % n) * (data.shape[2] + 1)
		unit[h_i: h_i+data.shape[1], w_i: w_i + data.shape[2] ] = data[i]
	return unit


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(20):
		loss_arr = []
		for _ in xrange(1000):
			train_x, trian_y = dataSets.train.next_batch(batch_size)
			_, loss_ = sess.run([train_op, loss], feed_dict={x:train_x, y:trian_y})
			loss_arr.append(loss_)
		print 'epoch: %d   loss: %.6f' % (i+1, np.array(loss_arr).mean())

	# 获取权重
	weights_ = sess.run(weights)
	weights_iamges = max_activate_weight(weights_.transpose([1, 0]))
	weights_iamges = weights_iamges.reshape([-1, crop_size, crop_size])
	one_image = images2one(weights_iamges, padval=0.0)
	misc.imsave('weights.png', one_image)





