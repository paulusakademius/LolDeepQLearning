import numpy as np
from PIL import ImageGrab
import cv2
from Press.KeyPress import PressKey,ReleaseKey,Q,W,E,R
import time


def process_image(image):
    originalImage = image
    processedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processedImage = cv2.resize(processedImage,None,fx=0.5,fy=0.5,interpolation= cv2.INTER_LINEAR)
    return processedImage

def main():
    while(True):

        printscreen = np.array(ImageGrab.grab(bbox=(0,0,1920,1080)))
        cv2.imshow('window',process_image(printscreen))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        PressKey(W)
        time.sleep(1)
        ReleaseKey(W)
        

if __name__ == "__main__":
    main()        