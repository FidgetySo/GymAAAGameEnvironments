#General Modules
import gym
from gym import spaces
import numpy as np

#Input Related Modules
import pydirectinput
import scipy.interpolate

#Screenshot Module
import d3dshot
d = d3dshot.create(capture_output="numpy")
print(d.displays)
d.display = d.displays[0]




#Image Modules
from skimage import measure
import cv2


import time

#Object Dectection Code
CONFIDENCE_THRESHOLD = 0.12
NMS_THRESHOLD = 0.12
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

def get_reward(image, frame, choice):
    reward = 0
    min_white=np.array([ 190, 190, 190])
    max_white=np.array([ 240, 240, 240])

    health_dst=cv2.inRange(frame , min_white , max_white)
    number_health =cv2.countNonZero(health_dst)
    health = number_health / 157
    try:
        previous_health = health
    except:
        previous_health = 1.0
    if health == 0:
        reward -= 1.5
    elif health < previous_health:
        reward -= previous_health - health * 3.75
    elif previous_health == health:
        reward -= 0.005
    #Crop to center of image to get if looking at enemy
    center = crop_center(image, 8, 8)
    min_blue=np.array([ 0, 255, 0])
    max_blue=np.array([ 0, 255, 0])
    blue_dst=cv2.inRange(center , min_blue , max_blue)
    blue =cv2.countNonZero(blue_dst)
    choice_list = list(choice)
    action_2 = choice_list[1]
    if blue >= 1 and action_2 == 1:
        reward += 6
        #print("Enemy hit")
    elif action_2 == 1 and blue == 0:
        reward -= .625
    return reward
def do_action(choice):
    #Make one scalar per action catergory of actions
    choice_list = list(choice)
    action_1 = choice_list[0]
    action_2 = choice_list[1]
    action_3 = choice_list[2]
    
    #Mouse Actions
    if action_1 == 1:
        move(x=20, y=0)
    elif action_1 == 2:
        move(x=-20, y=0)
    elif action_1 == 3:
        move(x=0, y=20)
    elif action_1 == 4:
        move(x=0, y=-20)
    elif action_1 == 5:
        pass
    
    #Shooting Actions
    if action_2 == 1:
        pydirectinput.mouseDown(button="left")
        time.sleep(0.015)
        pydirectinput.mouseUp(button="left")
    elif action_2 == 2:
        pass
    #Movement actions
    if action_3 == 1:
        pydirectinput.keyDown('shift')
        pydirectinput.keyDown('w')
    elif action_3 == 2:
        pydirectinput.keyUp('shift')
        pydirectinput.keyUp('w')
    elif action_3 == 2:
        pydirectinput.keyDown('space')
        time.sleep(0.015)
        pydirectinput.keyUp('space')
    elif action_3 == 4:
        pass
class CustomEnv(gym.Env):
    def __init__(self):
        # Initialize Default Gym Varibles
        self.num_envs = 1
        self.action_space = spaces.MultiDiscrete([ 5, 2, 4])
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(135, 240, 1), dtype=np.uint8)
    def reset(self):
        # Reset Env/Get New Screenshot
        image = d.screenshot()
        image = cv2.resize(image,(240, 135), interpolation = cv2.INTER_CUBIC).astype(np.uint8)   
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape((135, 240, 1))
        return image

    def step(self, action):
        do_action(action)
        image = d.screenshot()
        obs = cv2.resize(image,(320, 180), interpolation = cv2.INTER_CUBIC)
        obs = prediction(obs)
            
        reward_frame = image[ 1001:1001 + 1 , 894:894 + 160 , : ]
            
        reward = get_reward(obs, reward_frame, action)
        #Done Check via ssim score    
        frame=image[ 0:0 + 38 , 0:0 + 93 , : ]
        s = measure.compare_ssim(org, frame, multichannel = True)
        done = s == 1
        if done:
            # Press Space to start new roud/life
            pydirectinput.keyUp('space')
            time.sleep(0.05)
            pydirectinput.keyDown('space')
            time.sleep(1)
        obs = cv2.resize(obs,(240, 135), interpolation = cv2.INTER_CUBIC).astype(np.uint8)
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # Reshape for network
        obs = obs.reshape((135, 240, 1))
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        #Not Much can do here
        pass
