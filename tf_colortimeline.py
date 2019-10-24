import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
filename = 'buttercup.mp4'
batch_size = 128

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

flat_x = tf.reshape(X, (-1,360*640,3))
kr = tf.reduce_mean(X[:,:,2], 0) # batch num, pixel, rgb channel
kg = tf.reduce_mean(X[:,:,1], 0)
kb = tf.reduce_mean(X[:,:,0], 0)
#y = tf.cast(tf.stack([kr, kg, kb], axis=-1), tf.int16)
y = tf.cast(tf.reduce_mean(X, 0), tf.int16)
rank = tf.rank(y)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	init.run()
	num_frames = count_frames(filename)
	print('Number of frames in video:',num_frames)
	cap = cv2.VideoCapture(filename)
	frames = []
	outputs = []
	num_batches = int(np.ceil(num_frames/batch_size))
	print(num_batches)
	batch = 0
	i = 0
	for batch in tqdm(range(num_batches)):
		for k in range(batch_size):
			ret, frame = cap.read()
			i+=1
			if not ret:
				break
			frames.append(frame)
			if len(frames) % batch_size == 0:
				batch+=1
				#print('Rank:', rank.eval(feed_dict={X: frames}))
				#print('Batch: {} of {}'.format(batch, num_batches))
				#print(rank.eval(feed_dict={X: frames}))
				result = y.eval(feed_dict={X: frames})
				
				outputs.append([r for r in result])
				#print(result.shape)
				frames = []
	outputs.append(y.eval(feed_dict={X: frames}))
	outputs = np.array(outputs)
	picture = []
	print('Num frames read:', i)
	print(outputs.shape)
	#for batch in outputs:
	#	picture.append([x[0],x[1],x[2]])
	print('Turning array into two dimensional picture...')
	print(len(picture))
	#picture = np.array([picture for x in tqdm(range(int(.3*num_frames)))])
	#print('Give the program a second as matplotlib loads the image...')
	
	plt.imshow(picture)
	
	plt.show()
cap.release()