import tkinter as tk
import cv2
import sys
import numpy as np
import mediapipe as mp
import pandas as pd
sys.path.insert(0, '/Users/thomasli/Desktop/development/hackProject/GazeTracking/')
from gaze_tracking import GazeTracking
import time
import csv
import datetime
import os
from notifypy import Notify
from PIL import Image, ImageTk

def open_cv_to_tkinter(image):
    # Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    image = Image.fromarray(image)
    # Convert to PhotoImage
    image = image.resize((640, 360))
    image = ImageTk.PhotoImage(image)
    return image

def setup_variables():

    global start_time
    global cur_time
    global last_attention_time
    global last_distracted_time
    global start_distracted
    global end_distracted
    global gaze
    global cap
    global lost_attention_threshold
    global paying_attention_threshold
    global paying_attention
    global vertical_gaze_threshold
    global horizontal_gaze_threshold
    global user_id
    global last_notif_time
    global filename
    global recording

    start_time = int(time.time() * 1000) #Time variables
    cur_time = start_time
    last_attention_time = start_time
    last_distracted_time = start_time

    #Timestamp varisbles
    start_distracted = ''
    end_distracted = ''

    gaze = GazeTracking() #Init Gaze Tracking
    cap = cv2.VideoCapture(0) # Capture video

    lost_attention_threshold = 3000 #time distracted before being marked as not paying attention (ms)
    paying_attention_threshold = 500 #time of paying attention before being marked as paying attention (ms)
    paying_attention = True #boolean representing attention - true by default

    vertical_gaze_threshold = 0.8 #how low gaze can be before being marked as distracted
    horizontal_gaze_threshold = 0.2 #how off to side gaze can be before being marked as distracted

    user_id = 1 #User ID.

    last_notif_time = 0 #notification spacingo ut

    recording = False

def get_frame():

    global start_time
    global cur_time
    global last_attention_time
    global last_distracted_time
    global start_distracted
    global end_distracted
    global gaze
    global cap
    global lost_attention_threshold
    global paying_attention_threshold
    global paying_attention
    global vertical_gaze_threshold
    global horizontal_gaze_threshold
    global user_id
    global last_notif_time
    global filename

    ret, frame = cap.read()
    if not ret:
        return 0

    gaze.refresh(frame)

    vertical_ratio = gaze.vertical_ratio()
    horizontal_ratio = gaze.horizontal_ratio()
    cur_time = int(time.time() * 1000)
    frame_time = str(cur_time - start_time)

    if (paying_attention == False):
        if (cur_time - last_notif_time > 1000): 
            notification = Notify()
            notification.title = "Notice"
            notification.message = "Pay Attention!"
            notification.audio = "notif.wav"
            notification.send(block=False)
            last_notif_time = cur_time

    if (paying_attention):
        if (vertical_ratio != None and vertical_ratio < vertical_gaze_threshold and abs(0.5-horizontal_ratio) < horizontal_gaze_threshold):
            last_attention_time = cur_time
        elif (cur_time - last_attention_time > lost_attention_threshold):
            paying_attention = False
            start_distracted = cur_time
    else:
        if (vertical_ratio == None or vertical_ratio > vertical_gaze_threshold or abs(0.5-horizontal_ratio) > horizontal_gaze_threshold):
            last_distracted_time = cur_time
        elif (cur_time - last_distracted_time > paying_attention_threshold):
            paying_attention = True
            end_distracted = cur_time
            row = [1, start_distracted, end_distracted]
            with open(filename, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)

    print(str(vertical_ratio) + ", " + str(horizontal_ratio))
    print(str(vertical_gaze_threshold) + " - " + str(horizontal_gaze_threshold))
    print()
    return gaze.annotated_frame()

def setup_csv():
    global filename
    #create file for logging timestamps
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.getcwd() + '/logs/'
    filename = os.path.join(folder_path, f"log_{formatted_datetime}.csv")
    header = ['id', 'start_distracted', 'end_distracted']
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

def update_frame():

    global label
    tkinter_frame = open_cv_to_tkinter(get_frame())
    label.tk_image = tkinter_frame  # To prevent garbage collection
    global paying_attention
    status_text = ''
    if (paying_attention): status_text='Focused'
    else: status_text='Distracted'
    custom_font = ("Helvetica", 30)
    label.config(image=tkinter_frame, pady=20, text=status_text, font=custom_font, compound="top")
    
    global recording
    if (recording):
        label.after(5, update_frame)  # 10ms delay between frame updates

def start_recording():
    global recording
    if (recording == False):
        recording = True
        record_button.config(text="Recording...")
        setup_csv()
        update_frame()
    else:
        recording = False
        global cap
        global cv2
        global label
        cap.release()
        cv2.destroyAllWindows()
        exit()
        
def update_horizontal(value):
    global horizontal_gaze_threshold
    horizontal_gaze_threshold = float(value)/100.0

def update_vertical(value):
    global vertical_gaze_threshold
    vertical_gaze_threshold = float(value)/100.0

# Create the main window
root = tk.Tk()
root.title("FocusCam")

#Window Dimensions
window_width = 1920 
window_height = 1080
root.geometry(f"{window_width}x{window_height}")

#Setup variables
setup_variables()

# Create a Record button
record_button = tk.Button(root, text="Begin Session", width=40, height=4, command=start_recording)
record_button.pack()

global label
label = tk.Label(root)
label.pack()

padding1 = tk.Label(root, pady=10)

global horizontal_slider
global vertical_slider
horizontal_slider = tk.Scale(root, from_=0, to=100, orient="horizontal", length=300, command=update_horizontal, label="Horizontal Threshold")
vertical_slider = tk.Scale(root, from_=0, to=100, orient="horizontal", length=300, command=update_vertical, label="Vertical Threshold")
horizontal_slider.set(horizontal_gaze_threshold * 100)
vertical_slider.set(vertical_gaze_threshold * 100)
horizontal_slider.pack()
padding1.pack()
vertical_slider.pack()

# Run the GUI
root.mainloop()
cap.release()