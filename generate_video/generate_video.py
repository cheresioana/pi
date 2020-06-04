import cv2
import argparse
import os


dir_path = './results5'
ext = ".jpg"
output = "./ok10.mp4"

images = []
for f in os.listdir(dir_path):
	if f.endswith(ext):
		images.append(f)


image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape


fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output, fourcc, 4.0, (width, height))
i = 0;
while i < 450:
	image_path = dir_path + "/" + str(i) + ext;
	
	frame = cv2.imread(image_path)
	if frame is not None:	
		out.write(frame) # Write out frame to video

		cv2.imshow('video',frame)
		if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
			break
	i = i + 1;

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()
