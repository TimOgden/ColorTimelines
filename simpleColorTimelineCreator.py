import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

if __name__ == '__main__':
	filename, desired_width, desired_height = None, None, None
	try:
		filename = sys.argv[1]
		desired_width = int(sys.argv[2])
		desired_height = int(sys.argv[3])
	except:
		pass

	if filename is not None:
		num_frames = count_frames(filename)
		cap = cv2.VideoCapture(filename)
		avg_colors = []
		if desired_width:
			frequency = int(num_frames/desired_width)
		else:
			frequency = 1
		for i in tqdm(range(0,num_frames,frequency)):
			ret, frame = cap.read()
			#print(i, ret)
			# Our operations on the frame come here
			#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Display the resulting frame
			#cv2.imshow('frame',gray)
			#if cv2.waitKey(1) & 0xFF == ord('q'):
			#	break
			if not ret:
				break
			avg_colors.append([np.mean(frame[:,:,2])/255., np.mean(frame[:,:,1])/255., 
					np.mean(frame[:,:,0])/255.])
			
		cap.release()
		if not desired_height:
			height = int(.3*len(avg_colors))
		else:
			height = desired_height
		avg_colors = np.array(avg_colors)
		picture = np.array([avg_colors for x in range(height)])
		print(picture.shape)
		plt.figure()
		plt.title('Color Timeline')
		seconds_from_frame = [int(frame_num/30) for frame_num in range(0,i,300)]
		plt.xticks(range(0,i,300), seconds_from_frame)
		plt.xlabel('Time into video (s)')
		plt.yticks([])
		plt.imshow(picture)
		plt.show()