import numpy as np
from PIL import ImageGrab
import cv2
import time




def process_image(image):
    originalImage = image
    processedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processedImage = cv2.resize(processedImage,None,fx=0.5,fy=0.5,interpolation= cv2.INTER_LINEAR)
    return processedImage


        
def getScreen(): #returns current Frame

    screen = np.array(ImageGrab.grab(bbox=(0,0,1920,1080)))
    process_image(screen)
    return screen


def getKillsNumber():
    kills = np.array(ImageGrab.grab(bbox=(1660,0,1688,28)))
    return kills

def getDeathsNumber():
    kills = np.array(ImageGrab.grab(bbox=(0,0,0,0)))

def getGoldNumber():
    kills = np.array(ImageGrab.grab(bbox=(0,0,0,0)))

def getCsNumber():
    kills = np.array(ImageGrab.grab(bbox=(0,0,0,0)))


if __name__ == '__main__':
    time.sleep(5)
    print(cv2.cvtColor(getKillsNumber(), cv2.COLOR_BGR2GRAY).shape)
    while(True):
        cv2.imshow('window',cv2.cvtColor(getKillsNumber(),cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break