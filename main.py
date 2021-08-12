import pyautogui
import numpy as np
import time
from PIL import Image
import cv2
import math
from directkeys import PressKey, ReleaseKey, W, A, S, D
import cv2
import tensorflow as tf
import keyboard

def releaseKeys():
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

model = tf.keras.models.load_model('raceModelEdges2')
paused = False
while True:
    if (keyboard.is_pressed('e') and paused):
        paused = False
        print('unpaused')
        time.sleep(1)
        ReleaseKey(W)
    elif(keyboard.is_pressed('e')):
        paused = True
        print('paused')
        time.sleep(1)
        releaseKeys()
        ReleaseKey(W)
    if(paused):
        time.sleep(1)
        continue
    PressKey(W)

    screen = pyautogui.screenshot()
    screen = screen.crop((100, 400, 1600, 750))
    screen = np.asarray(screen)
    resizedScreen = cv2.resize(screen, (160, 160))
    gray = cv2.cvtColor(resizedScreen, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    canny = cv2.Canny(blurred, 100, 200)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    """map = screen.crop((50, 875, 300, 1050))
    screen = screen.crop((100, 300, 1700, 1080))
    screen = np.asarray(screen)
    map = np.asarray(map)"""

    # applies a mask to the map so only the path shows
    """lower = np.array([100, 20, 180], dtype="uint8")
    upper = np.array([200, 120, 255], dtype="uint8")
    mask = cv2.inRange(map, lower, upper)
    masked = cv2.bitwise_and(map, map, mask=mask)"""

    # resizes the components and stacks them
    """resizedScreen = cv2.resize(screen, (160, 160))
    resizedMap = cv2.resize(map, (60, 60))
    resizedScreen[100:, :60] = resizedMap  # puts map on the bottom right of image"""

    prediction = model.predict(canny[np.newaxis, ...])
    classes = ['a', 'd', 'none']
    keyToPress = classes[np.argmax(prediction)]
    print(keyToPress)
    releaseKeys()

    if(keyToPress == 'none'):
        continue
    elif(keyToPress == 'a'):
        PressKey(A)
    elif(keyToPress == 'd'):
        PressKey(D)