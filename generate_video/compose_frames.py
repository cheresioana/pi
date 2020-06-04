matrix = np.loadtxt("model_predict3.txt", usecols=range(6))
print(matrix)
print(len(matrix))

# Commented out IPython magic to ensure Python compatibility.
import cv2
# %ls

from google.colab.patches import cv2_imshow

frame_x = 480
frame_y = 640
count = 0
out_folder = "./results5/"
img_type = "sketch"
for item in matrix:
  blank_image = np.zeros((480,640,4), np.uint8)
  blank_image[:, :, :] = (255, 255, 255, 0)
  
  tom_nr = (int)(item[2])
  tom_x = (int)(item[0])
  tom_y = (int)(item[1])
  jerry_nr = (int)(item[5])
  jerry_x = (int)(item[3])
  jerry_y = (int)(item[4])
  img_jerry = None
  img_tom = None
  if tom_nr != -1:
    img_tom = cv2.imread("./out/"+ str(tom_nr) + "tom" + img_type + ".png", cv2.IMREAD_UNCHANGED)
  if jerry_nr != -1:  
    img_jerry = cv2.imread("./out/"+ str(jerry_nr) + "jerry"+ img_type + ".png", cv2.IMREAD_UNCHANGED)
  #cv2_imshow(blank_image[:, :, :3])
  if (img_tom is not None):
    try:
      tom_x_end = tom_x + img_tom.shape[0]
      tom_y_end = tom_y + img_tom.shape[1]
      
    
      if (tom_y_end > frame_y):
        tom_y = tom_y - (tom_y_end - frame_y)
        tom_y_end = tom_y + img_tom.shape[1]
      if (tom_x_end > frame_x):
        tom_x = tom_x - (tom_x_end - frame_x)
        tom_x_end = tom_x + img_tom.shape[0]
      r = 0
      for i in range(tom_x, tom_x_end):
        c = 0
        for j in range(tom_y, tom_y_end):
          if img_tom[r, c, 3] == 255:
            blank_image[ i, j] = img_tom[r, c]
          c = c + 1;
        r = r + 1
      #cv2_imshow(blank_image) 
      #cv2_imshow(blank_image)
    except:
      print("aia e")
  if (img_jerry is not None):
    try:    
      jerry_x_end = jerry_x + img_jerry.shape[0]
      jerry_y_end = jerry_y + img_jerry.shape[1]
      if (jerry_y_end > frame_y):
        jerry_y = jerry_y - (jerry_y_end - frame_y)
        jerry_y_end = jerry_y + img_jerry.shape[1]
      if (jerry_x_end > frame_x):
        jerry_x = jerry_x - (jerry_x_end - frame_x)
        jerry_x_end = jerry_x + img_jerry.shape[0]
      r = 0
      for i in range(jerry_x, jerry_x_end):
        c = 0
        for j in range(jerry_y, jerry_y_end):
          if img_jerry[r, c, 3] == 255:
            blank_image[ i, j] = img_jerry[r, c]
          c = c + 1;
        r = r + 1
      #cv2_imshow(blank_image) 
    except:
      print("aia e")
  if (img_jerry is not None or img_tom is not None):
    #cv2_imshow(blank_image)
    cv2.imwrite(out_folder + str(count) + ".jpg", blank_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
  count = count + 1
  if count % 10 == 0:
    print(count)
print("gata")