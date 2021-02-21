#General Modules
import sys
import gym
from gym import spaces
import numpy as np

#Input Related Modules
import pydirectinput
import scipy.interpolate

#Screenshot Module
import d3dshot
d = d3dshot.create(capture_output="numpy")
d.display = d.displays[0]



#Image Modules
from skimage import measure
import cv2


import time



#Object Dectection Code
CONFIDENCE_THRESHOLD = 0.195
NMS_THRESHOLD = 0.28
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 180), scale=1/255)                                     
def prediction(image):

    image1 = cv2.UMat(image)

    classes, scores, boxes = model.detect(image1, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for (classid, score, box) in zip(classes, scores, boxes):
        if classid[0] == 0:
            image = cv2.rectangle(image, box, (0, 255, 0), cv2.FILLED)
    return image

#End Of Object Dectection

#Get Done Comparision Image
org = cv2.imread("org.jpg").astype(np.uint8)
pydirectinput.FAILSAFE = False

import ctypes
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


def left_click():
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    time.sleep(0.005)
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0004, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def move(x=None, y=None, duration=0.005, absolute=False, interpolate=False, **kwargs):
    if (interpolate):
        print("mouse move {}".format(interpolate))
        current_pixel_coordinates = win32api.GetCursorPos()
        if interpolate:
            current_pixel_coordinates = win32api.GetCursorPos()
            start_coordinates = _to_windows_coordinates(*current_pixel_coordinates)

            end_coordinates = _to_windows_coordinates(x, y)
            print("In interpolate")
            coordinates = _interpolate_mouse_movement(
                start_windows_coordinates=start_coordinates,
                end_windows_coordinates=end_coordinates
            )
            print(coordinates)
        else:
            coordinates = [end_coordinates]

        for x, y in coordinates:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(x, y, 0, (0x0001 | 0x8000), 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    else:
        x = int(x)
        y = int(y)

        coordinates = _interpolate_mouse_movement(
            start_windows_coordinates=(0, 0),
            end_windows_coordinates=(x, y)
        )

        for x, y in coordinates:
            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.mi = MouseInput(x, y, 0, 0x0001, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
            

def _to_windows_coordinates(x=0, y=0):
    display_width = win32api.GetSystemMetrics(0)
    display_height = win32api.GetSystemMetrics(1)

    windows_x = (x * 65535) // display_width
    windows_y = (y * 65535) // display_height

    return windows_x, windows_y

def _interpolate_mouse_movement(start_windows_coordinates, end_windows_coordinates, steps=20):
    x_coordinates = [start_windows_coordinates[0], end_windows_coordinates[0]]
    y_coordinates = [start_windows_coordinates[1], end_windows_coordinates[1]]

    if x_coordinates[0] == x_coordinates[1]:
        x_coordinates[1] += 1

    if y_coordinates[0] == y_coordinates[1]:
        y_coordinates[1] += 1

    interpolation_func = scipy.interpolate.interp1d(x_coordinates, y_coordinates)

    intermediate_x_coordinates = np.linspace(start_windows_coordinates[0], end_windows_coordinates[0], steps + 1)[1:]
    coordinates = list(map(lambda x: (int(round(x)), int(interpolation_func(x))), intermediate_x_coordinates))
    return coordinates
def crop_center(img,cropx,cropy):
    y, x, _= img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, starty:starty + cropy, :]
reward_addition = 0
def calc_reward_addition(done):
    global reward_addition
    if done:
        reward_addition = 0
        return 0
    else:
        reward_addition = reward_addition + 0.005
        reward = min((0.01 + reward_addition), 1)
        return reward
def get_reward(image, frame, choice, done):
    reward = 0
    min_white=np.array([ 190, 190, 190])
    max_white=np.array([ 240, 240, 240])

    health_dst=cv2.inRange(frame , min_white , max_white)
    number_health =cv2.countNonZero(health_dst)
    health = number_health / 157
    health = health * 100
    try:
        previous_health = health
    except:
        print("uh oh")
        previous_health = 100.0
    if health < previous_health:
        reward -= min((previous_health - health), 20)
    elif previous_health == health:
        if done:
            reward -= 75
        if not done:
            reward -= calc_reward_addition(done)
    #Crop to center of image to get if looking at enemy
    center = crop_center(image, 24, 24)
    min_blue=np.array([ 0, 240, 0])
    max_blue=np.array([ 15, 255, 15])
    blue_dst=cv2.inRange(center , min_blue , max_blue)
    blue =cv2.countNonZero(blue_dst)
    choice_list = list(choice)
    action_2 = choice_list[1]
    if blue >= 1 and action_2 == 0:
        reward += 100 
        print("Enemy hit")
    elif action_2 == 0 and blue == 0:
        reward -= 5
    return reward, health
def do_action(choice):
    #Make one scalar per action catergory of actions
    choice_list = list(choice)
    action_1 = choice_list[0]
    action_2 = choice_list[1]
    action_3 = choice_list[2]
    #Mouse Actions
    if action_1 == 0:
        move(x=20, y=0)
    elif action_1 == 1:
        move(x=-20, y=0)
    elif action_1 == 2:
        move(x=0, y=20)
    elif action_1 == 3:
        move(x=0, y=-20)
    elif action_1 == 4:
        pass
    
    #Shooting Actions
    if action_2 == 0:
        left_click()
    elif action_2 == 1:
        pass
    #Movement actions
    if action_3 == 0:
        pydirectinput.keyDown('shift')
        time.sleep(0.005)
        pydirectinput.keyDown('w')
    elif action_3 == 1:
        pydirectinput.keyUp('shift')
        time.sleep(0.005)
        pydirectinput.keyUp('w')
    elif action_3 == 2:
        pass
class CustomEnv(gym.Env):
    def __init__(self):
        # Initialize Default Gym Varibles
        self.num_envs = 1
        self.action_space = spaces.MultiDiscrete([ 5, 2, 3])
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(45, 80, 3), dtype=np.uint8)
    def reset(self):
        # Reset Env/Get New Screenshot
        image = d.screenshot()
        image = cv2.resize(image.astype(np.uint8),(80, 45), interpolation = cv2.INTER_NEAREST) 
        #image = np.reshape(image, (45,80, 1))
        return image

    def step(self, action):
        do_action(action)
        image = d.screenshot()
        obs = cv2.resize(image.astype(np.uint8),(320, 180), interpolation = cv2.INTER_NEAREST)
        obs = prediction(obs)
            
        reward_frame = image[ 1001:1001 + 1 , 894:894 + 160 , : ]
        
        frame=image[ 0:0 + 38 , 0:0 + 93 , : ]
        s = measure.compare_ssim(org, frame, multichannel = True)
        done = s == 1    
        
        
        reward, health = get_reward(obs, reward_frame, action, done)
        #Done Check via ssim score    
        if done:
            # Press Space to start new roud/life
            pydirectinput.keyUp('space')
            time.sleep(0.03)
            pydirectinput.keyDown('space')
            time.sleep(1)
        obs = cv2.resize(obs.astype(np.uint8),(80, 45), interpolation = cv2.INTER_NEAREST)
        # Reshape for network
        #obs = np.reshape(obs, (45,80, 1))
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        #Not Much can do here
        pass
