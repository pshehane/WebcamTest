import cv2
import numpy as np

cv2.namedWindow("preview")
maxCounter = 180000
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 4

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

def testWebcam(requested_width, requested_height):
    print("Testing %d x %d" % (requested_width, requested_height))
    counter = 0
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vc.get(cv2.CAP_PROP_FPS))
    waitTime = int(1000.0 / float(fps))
    print("FPS %f, so wait: %d" % (fps, waitTime))
    refocusCount = 0
    badFrameCount = 0
    error = 0.0
    if ((requested_width != width) or (requested_height != height)):
        print("Unable to support requested width height")
        vc.release()
        return

    while rval:
        cv2.putText(frame, "%d x %d @ %f Count: %d out of %d - %f" % (width, height, fps, counter, maxCounter, error), org, font, fontScale, color, thickness)
        cv2.imshow("preview", frame)
        prevFrame = frame
        rval, frame = vc.read()
        gray_img1 = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        error = mse(gray_img1, gray_img2)
        if (error > 10.0):
            badFrameCount += 1
        elif (error > 1.0):
            refocusCount += 1

        key = cv2.waitKey(waitTime)
        if key == 27: # exit on ESC
            break
        
        counter = counter + 1
        if counter > maxCounter:
            break
    print("Out of %d frames, got %d bad frames and %d refocuses" % (counter, badFrameCount, refocusCount))

    vc.release()
    cv2.destroyWindow("preview")


#testWebcam(640,360)
#testWebcam(640,480)
#testWebcam(800,600)
testWebcam(1280,720)
#testWebcam(1600,1200)
