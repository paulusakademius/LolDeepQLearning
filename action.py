from packages.KeyPress import PressKey,ReleaseKey,Q,W,E,R

def step(action):
    if action==0:
        PressKey(Q)
    if action==1:
        PressKey(W)
    if action==2:
        PressKey(E)
    if action==3:
        PressKey(R)