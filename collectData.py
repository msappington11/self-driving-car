import numpy as np
import cv2
import keyboard
import pyautogui
import time
import tensorflow as tf
import pathlib
from PIL import Image

def collectData():
    time.sleep(5)
    counter = 0
    while True:
        key = 'none'
        if keyboard.is_pressed('a'):
            key = 'a'
        if keyboard.is_pressed('d'):
            key = 'd'
        elif(keyboard.is_pressed('esc')):
            break
        #if key == 'none':
            #continue

        # takes a screenshot and separates the map. converts to numpy array
        screen = pyautogui.screenshot()
        #map = screen.crop((50, 875, 300, 1050))
        screen = screen.crop((100, 400, 1600, 750))
        screen = np.asarray(screen)
        #map = np.asarray(map)

        # resizes the components and stacks them/saves them
        resizedScreen = cv2.resize(screen, (160, 160))
        #resizedMap = cv2.resize(map, (60, 60))
        #resizedColor = np.vstack((resizedScreen, resizedMap))
        #resizedScreen[100:, :60] = resizedMap # puts map on the bottom right of image
        #cv2.imwrite(r'C:\Users\Scot\Desktop\Nerd Stuff\GTADataCollection\race data\raceColor\{}\sample1_{}.png'.format(key, counter), resizedScreen)

        # applies a mask to the map so only the path shows
        '''lower = np.array([100, 20, 180], dtype="uint8")
        upper = np.array([200, 120, 255], dtype="uint8")
        mask = cv2.inRange(map, lower, upper)
        maskedMap = cv2.bitwise_and(map, map, mask=mask)'''

        # puts the masked map on the screen then applies edge detection
        #resizedMaskedMap = cv2.resize(maskedMap, (60, 60))
        #resizedScreen[100:, :60] = resizedMaskedMap  # puts masked map on the bottom right of image
        gray = cv2.cvtColor(resizedScreen, cv2.COLOR_RGB2GRAY)
        blurred = cv2.medianBlur(gray, 3)
        canny = cv2.Canny(blurred, 100, 200)
        cv2.imwrite(r'C:\Users\Scot\Desktop\Nerd Stuff\GTADataCollection\race data\raceEdges\{}\sample4_{}.png'.format(key, counter), canny)
        counter += 1

def makeNetwork():
    data_dir = (r'C:\Users\Scot\Desktop\Nerd Stuff\GTADataCollection\race data\raceEdges')

    # breaking up the data into training and validation
    BATCH_SIZE = 32
    IMG_HEIGHT = 160
    IMG_WIDTH = 160

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,  # input data
        validation_split=0.2,  # 20% validation, 10% test
        subset="training",
        seed=123,  # optional randomizer value
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,  # input data
        validation_split=0.2,  # 20% validation
        subset="validation",
        seed=123,  # optional randomizer value
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    print(train_ds)

    # prepares the images by putting them into caches (performance boost)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE))  # first layer. 32 filters in 3x3 size. input shape must be given (image is 32x32x3 for rgb)
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))  # pools data from 2x2 area with stride of 2 (no overlap for pooling)
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))  # another layer with 64 filters in 3x3 area
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))  # pools again
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))  # another layer with 64 filters in 3x3 area
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))  # 64 nodes
    model.add(tf.keras.layers.Dense(3))  # number of outputs

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_ds, epochs=4,
                        validation_data=val_ds)
    model.save(r'C:\Users\Scot\Desktop\Nerd Stuff\GTADriving\raceModelEdges25')

makeNetwork()
#collectData()