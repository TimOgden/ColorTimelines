import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
filename = 'buttercup.mp4'
batch_size = 64

def count_frames(path, override=False):
	# grab a pointer to the video file and initialize the total
	# number of frames read
	video = cv2.VideoCapture(path)
	total = 0

	total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))



	# release the video file pointer
	video.release()

	# return the total number of frames in the video
	return total

X = tf.placeholder(tf.float32, name='X')
kr = tf.reduce_mean(X[:,:,:,2])
kg = tf.reduce_mean(X[:,:,:,1])
kb = tf.reduce_mean(X[:,:,:,0])
y = tf.cast(tf.stack([kr, kg, kb], axis=0), tf.int16)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	init.run()
	num_frames = count_frames(filename)
	print('Number of frames in video:',num_frames)
	cap = cv2.VideoCapture(filename)
	frames = []
	outputs = []
	while(True):
		ret, frame = cap.read()
		if not ret:
			break
		frames.append(frame)
		if len(frames) % batch_size == 0:
			outputs.append(y.eval(feed_dict={X: frames}))
			frames = []
	outputs.append(y.eval(feed_dict={X: frames}))
	picture = np.array([outputs for x in range(int(len(outputs)*.3))])
	plt.imshow(picture)
	plt.show()
cap.release()