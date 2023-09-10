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
from PIL import Image, ImageTk, ImageDraw

def add_corners(im, rad):
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2 - 1, rad * 2 - 1), fill=255)
    alpha = Image.new('L', im.size, 255)
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im

def open_cv_to_tkinter(image):
    # Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    image = Image.fromarray(image)
    image = add_corners(image, 30)
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
    global face_cascade

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    global face_cascade

    ret, frame = cap.read()
    if not ret:
        print('Error in retrieving frame')
        exit()

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    faces = sorted(faces, key=lambda x: x[2] * x[3])
    if (len(faces) < 1):
        return frame
    x, y, w, h = faces[-1]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_color = frame[y:y + h, x:x + w]
    gaze.refresh(roi_color)

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
    print(str(lost_attention_threshold) + " - " + str(paying_attention_threshold))
    print()
    return frame
    # return gaze.annotated_frame()

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
        record_button.config(text="Stop Recording")
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

def update_lost_attention_threshold(value):
    global lost_attention_threshold
    lost_attention_threshold = int(value)

def update_paying_attention_threshold(value):
    global paying_attention_threshold
    paying_attention_threshold = int(value)

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

frame1 = tk.Frame(root)
frame2 = tk.Frame(root)
frame3 = tk.Frame(root)
frame4 = tk.Frame(root)

global horizontal_slider
global vertical_slider
global lost_attention_slider
global pay_attention_slider

horizontal_slider = tk.Scale(frame1, from_=0, to=100, orient="horizontal", length=300, command=update_horizontal, label="Horizontal Threshold")
vertical_slider = tk.Scale(frame2, from_=0, to=100, orient="horizontal", length=300, command=update_vertical, label="Vertical Threshold")
lost_attention_slider = tk.Scale(frame3, from_=0, to=10000, orient="horizontal", length=300, command=update_lost_attention_threshold, label="Lost Attention Threshold (ms)")
pay_attention_slider = tk.Scale(frame4, from_=0, to=10000, orient="horizontal", length=300, command=update_paying_attention_threshold, label="Pay Attention Threshold (ms)")

horizontal_slider.set(horizontal_gaze_threshold * 100)
vertical_slider.set(vertical_gaze_threshold * 100)
lost_attention_slider.set(lost_attention_threshold)
pay_attention_slider.set(paying_attention_threshold)

horizontal_slider.pack(side="top", padx=10, pady=10)
# padding1.pack()
vertical_slider.pack(side="bottom", padx=10, pady=10)
# padding2.pack()
lost_attention_slider.pack(side="top", padx=10, pady=10)
# padding3.pack()
pay_attention_slider.pack(side="bottom", padx=10, pady=10)

frame1.pack(side="left", padx=30)
frame2.pack(side="left", padx=30)
frame3.pack(side="right", padx=30)
frame4.pack(side="right", padx=30)

# Run the GUI
root.mainloop()
cap.release()