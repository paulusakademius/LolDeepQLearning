from packages.KeyPress import PressKey,ReleaseKey,Q,W,E,R,B
from packages.ScreenGrab import getScreen
from packages.reward import getReward
import mouse

def step(action):  #takes given action and returns new frame and reward
    if action==0:
        PressKey(Q)
        ReleaseKey(Q)
    if action==1:
        PressKey(W)
        ReleaseKey(W)
    if action==2:
        PressKey(E)
        ReleaseKey(E)
    if action==3:
        PressKey(R)
        ReleaseKey(R)
    if action==4:
        mouse.move(20,0,False,0.1)
    if action==5:
        mouse.move(0,20,False,0.1)
    if action==6:
        mouse.click('right')
    if action==7:
        PressKey(B)
        ReleaseKey(B)

    return getScreen(), getReward()