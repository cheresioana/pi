import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import functools 
from google.colab.patches import cv2_imshow
from imutils import contours
import cv2

options = {
    'model' : 'cfg/tiny-yolo-voc-2c.cfg',
    'load' : 29500,
    'threshold': 0.005,
    'gpu': 0.8
}

tfnet = TFNet(options)
colors = [(0, 0, 255), (255, 255, 200)]

def test():
  imgcv = cv2.imread("./test/input/154.jpg")
  result = tfnet.return_predict(imgcv)

  tom_array = [x for x in result if x['label']=='tom']
  jerry_array = [x for x in result if x['label']=='jerry']

  if len(tom_array) > 0:
    tom = functools.reduce(lambda a,b: a if (a['confidence'] > b['confidence']) else b, tom_array);
    tom_crop = imgcv[tom['topleft']['y'] :tom['bottomright']['y'], tom['topleft']['x'] :tom['bottomright']['x']]
    tom_rect = (tom['topleft']['x'], tom['topleft']['y'] ,tom['bottomright']['x'] - tom['topleft']['x'] ,tom['bottomright']['y'] - tom['topleft']['y'])

  if len(jerry_array) > 0:
    jerry = functools.reduce(lambda a,b: a if (a['confidence'] > b['confidence']) else b, jerry_array);
    jerry_rect = (jerry['topleft']['x'], jerry['topleft']['y'] ,jerry['bottomright']['x'] - jerry['topleft']['x'] ,jerry['bottomright']['y'] - jerry['topleft']['y'])
    jerry_crop = imgcv[jerry['topleft']['y']  :jerry['bottomright']['y'] , jerry['topleft']['x'] :jerry['bottomright']['x'] ]

    gray = cv2.cvtColor(jerry_crop, cv2.COLOR_BGR2GRAY)
    cv2_imshow(gray)
    print(gray.shape)
    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

    th3 = cv2.bitwise_not(th3)
    cv2_imshow(th3)

    blur = cv2.medianBlur(th3, 3)
    cv2_imshow(blur)

    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(blur,kernel,iterations = 1)
    cv2_imshow(erosion)

    kernel = np.ones((4,4),np.uint8)
    if (gray.size < 32400):
      kernel = np.ones((5,5),np.uint8)

    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    cv2_imshow(dilation)


    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=lambda a:a.size)
    
    cv2.drawContours(jerry_crop, [max_contour], 0, (50,50,50), 2)
    cv2_imshow(jerry_crop)
    mask = np.zeros_like(gray) 
    cv2.drawContours(mask, [max_contour], 0, 255, -1)
    
    out = np.zeros_like(jerry_crop)
    
    out[mask == 255] = jerry_crop[mask == 255]
    cv2_imshow(out)
   
    rgba = cv2.cvtColor(out, cv2.COLOR_RGB2RGBA)
    #print(rgba)
    rgba[:,:, 3] = 0
    rgba[mask==255, 3] = 255
    cv2_imshow(rgba)
    cv2.imwrite("./out/1color.png", rgba)
    kernel = np.ones((3,3),np.uint8)
    if (gray.size < 32400):
      kernel = np.ones((4,4),np.uint8)
    
    sketchy = cv2.erode(dilation,kernel,iterations = 1)
    sketchy = cv2.bitwise_not(sketchy)
    mask = np.zeros_like(gray) 
    cv2.drawContours(mask, [max_contour], 0, 255, -1)
    out = np.zeros_like(sketchy)
    out[mask == 255] = sketchy[mask == 255]
    cv2.imwrite("./out/1sketchy.png", out)
    cv2_imshow(out)

  def process_image(img, number):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.bitwise_not(th3)
    blur = cv2.medianBlur(th3, 3)
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(blur,kernel,iterations = 1)
    kernel = np.ones((4,4),np.uint8)

    if (gray.size < 32400):
      kernel = np.ones((5,5),np.uint8)
    
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=lambda a:a.size)
    cv2.drawContours(img, [max_contour], 0, (50,50,50), 2)
    mask = np.zeros_like(gray) 
    cv2.drawContours(mask, [max_contour], 0, 255, -1)
    
    out = np.zeros_like(img)
    out[:, :] = (255, 255, 255)
    out[mask == 255] = img[mask == 255]
    rgba_color = cv2.cvtColor(out, cv2.COLOR_RGB2RGBA)
    rgba_color[:,:, 3] = 0
    rgba_color[mask==255, 3] = 255
    rgb_color = out
    #cv2.imwrite("./out/" + str(number) + "color.png", rgba)
    
    
    kernel = np.ones((3,3),np.uint8)
    if (gray.size < 32400):
      kernel = np.ones((4,4),np.uint8)
    
    sketchy = cv2.erode(dilation,kernel,iterations = 1)
    sketchy = cv2.bitwise_not(sketchy)
    mask = np.zeros_like(gray) 
    cv2.drawContours(mask, [max_contour], 0, 255, -1)
    out = np.zeros_like(sketchy)
    out[mask == 255] = sketchy[mask == 255]
    rgba_sketch = cv2.cvtColor(out, cv2.COLOR_RGB2RGBA)
    rgba_sketch[:,:, 3] = 0
    rgba_sketch[mask==255, 3] = 255
    #cv2.imwrite("./out/" + str(number) + "sketchy.png", rgba)
    return (rgba_color, rgb_color, rgba_sketch)

def check_result(color, rgb_color, sketch, label, number, frame):
  cv2.imwrite("./out/" + str(number) + label + "color.png", color)
  cv2.imwrite("./out/" + str(number) + label + "sketch.png", sketch)


cap = cv2.VideoCapture("/content/gdrive/My Drive/desene/desene_proj/my_files/vid3.mp4")
count = 0
f = open("./data.txt", "w")
f.write("x_tom,y_tom,photo_tom,x_jerry,y_jerry,photo_jerry,pred_x_tom,pred_y_tom,pred_photo_tom,pred_x_jerry,pred_y_jerry,pred_photo_jerry\n")
s1 = None
s2 = None
while cap.isOpened() and count < 500:
    ret,frame = cap.read()
    detect_anything = 0;
    if count == 0:
      print(frame.shape)
    if ret :
      result = tfnet.return_predict(frame)
      tom_array = [x for x in result if x['label']=='tom']
      jerry_array = [x for x in result if x['label']=='jerry']
      tom_photo_number = -1;
      jerry_photo_number = -1;
      tom_corner = (-1, -1);
      jerry_corner = (-1, -1);
      if len(tom_array) > 0:
        detect_anything = 1;
        tom = functools.reduce(lambda a,b: a if (a['confidence'] > b['confidence']) else b, tom_array);
        tom_crop = frame[tom['topleft']['y'] :tom['bottomright']['y'], tom['topleft']['x'] :tom['bottomright']['x']]
        tom_corner = (tom['topleft']['x'], tom['topleft']['y'])
        rgba_color, rgb_color, rgba_sketch = process_image(tom_crop, count)
        check_result(rgba_color, rgb_color, rgba_sketch, "tom", count, frame)
        tom_photo_number = count

      if len(jerry_array) > 0:
        detect_anything = 1;
        jerry = functools.reduce(lambda a,b: a if (a['confidence'] > b['confidence']) else b, jerry_array);
        jerry_crop = frame[jerry['topleft']['y']  :jerry['bottomright']['y'] , jerry['topleft']['x'] :jerry['bottomright']['x'] ]
        jerry_corner = (jerry['topleft']['x'], jerry['topleft']['y'])
        rgba_color, rgb_color, rgba_sketch = process_image(jerry_crop, count)
        check_result(rgba_color, rgb_color, rgba_sketch, "jerry", count, frame)
        jerry_photo_number = count
      
      if detect_anything == 0:
        s1 = None
      elif s1 == None:
        s1 = str(tom_corner[0]) + "," + str(tom_corner[1]) + "," + str(tom_photo_number) + "," + str(jerry_corner[0]) + "," + str(jerry_corner[1]) + "," + str(jerry_photo_number);
      else:
        s2 = str(tom_corner[0]) + "," + str(tom_corner[1]) + "," + str(tom_photo_number) + "," + str(jerry_corner[0]) + "," + str(jerry_corner[1]) + "," + str(jerry_photo_number);
        f.write(s1 + "," + s2 + "\n");
        s1 = s2;
      count = count + 1

      
f.close()
print(count)
cap.release()
cv2.destroyAllWindows()