import csv
import os
import tkinter as tk
from tkinter import messagebox as mess
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from main import TrainImages

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)

def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess.showerror("Missing File", "Please contact support for missing files.")
        window.destroy()

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clear():
    if txt.winfo_exists():
        txt.delete(0, 'end')

def clear2():
    if txt2.winfo_exists():
        txt2.delete(0, 'end')

def TakeImages():
    try:
        if not all(widget.winfo_exists() for widget in [txt, txt2]):
            mess.showerror("Error", "UI elements not available. Please restart the application.")
            return

        Id = txt.get().strip()
        name = txt2.get().strip()
        role = role_var.get()

        if not Id.isdigit():
            mess.showerror("Invalid ID", "Enter a valid numeric ID.")
            return

        if not name.replace(" ", "").isalpha():
            mess.showerror("Invalid Name", "Enter a valid name (letters only).")
            return

        file_path = "StudentDetails/StudentDetails.csv"
        training_path = "TrainingImage/"
        
        if os.path.isfile(file_path):
            with open(file_path, 'r') as csvFile:
                reader = csv.reader(csvFile)
                next(reader) 
                for row in reader:
                    if row and row[1] == Id:
                        mess.showerror("Duplicate ID", f"ID {Id} is already registered.")
                        return

        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cam.isOpened():
            mess.showerror("Error", "Could not open camera.")
            return

        assure_path_exists("StudentDetails/")
        assure_path_exists(training_path)

        sampleNum = 0
        face_detected = False
        min_face_size = 100  

        while sampleNum < 50: 
            ret, img = cam.read()
            if not ret:
                continue
                
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    face_width = x2 - x1
                    face_height = y2 - y1
                    
                  
                    if (x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and 
                        x2 <= img.shape[1] and y2 <= img.shape[0] and
                        face_width >= min_face_size and face_height >= min_face_size):
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        face_img = frame_rgb[y1:y2, x1:x2]
                        
                        if face_img.size > 0:
                            try:
                           
                                face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                                face_gray = cv2.equalizeHist(face_gray)
                                
                            
                                face_gray = cv2.resize(face_gray, (160, 160))
                                
                              
                                save_path = f"{training_path}{name}.{Id}.{sampleNum+1}.jpg"
                                if cv2.imwrite(save_path, face_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                                    sampleNum += 1
                                    face_detected = True
                                    cv2.putText(img, f"Samples: {sampleNum}/50", (10, 30),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Face processing error: {e}")
                                continue
            
            cv2.imshow('Taking Images - Press Q to cancel', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    
        cam.release()
        cv2.destroyAllWindows()

        if not face_detected or sampleNum == 0:
            mess.showerror("Error", "No valid faces detected during capture. Please try again with better lighting.")
          
            if sampleNum > 0:
                for i in range(1, sampleNum+1):
                    try:
                        os.remove(f"{training_path}{name}.{Id}.{i}.jpg")
                    except:
                        pass
            return

        
        if not os.path.isfile(file_path):
            serial_no = 1
        else:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader) 
                serial_no = sum(1 for _ in reader) + 1

        with open(file_path, 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            if os.stat(file_path).st_size == 0: 
                writer.writerow(["SERIAL NO.", "ID", "NAME", "ROLE"])
            writer.writerow([serial_no, Id, name, role])

      
        mess.showinfo("Success", 
                     f"Successfully registered {name} (ID: {Id}) as {role}\n"
                     f"Serial No: {serial_no}\n"
                     f"Captured {sampleNum} high-quality face samples.")
        TrainImages()
        clear()
        clear2()

    except Exception as e:
        mess.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
        print(f"Error details: {e}")
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()

window = tk.Tk()
window.geometry("700x500")
window.title("Attendance System")

try:
    bg_img = Image.open("img.png").resize((800, 600))
    bg_photo = ImageTk.PhotoImage(bg_img)
    bg_label = tk.Label(window, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)
except Exception as e:
    print(f"Background image error: {e}")
    window.configure(bg="#f0f0f0")

title_label = tk.Label(window, text="New Registration", font=("comic", 16, "bold"), fg="black", bg="#f0f0f0")
title_label.place(relx=0.5, rely=0.15, anchor="center")

lbl = tk.Label(window, text="Enter ID", font=('comic', 12), fg="black", bg="#f0f0f0")
lbl.place(relx=0.3, rely=0.3, anchor="center")

txt = tk.Entry(window, font=('comic', 12), bg="#ffffff", fg="black", bd=2, relief="solid")
txt.place(relx=0.51, rely=0.3, anchor="center", width=200)

clear_btn1 = tk.Button(window, text="Clear", command=clear, bg="#DC3545", fg="white", font=('comic', 10))
clear_btn1.place(relx=0.7, rely=0.3, anchor="center")

lbl2 = tk.Label(window, text="Enter Name", font=('comic', 12), fg="black", bg="#f0f0f0")
lbl2.place(relx=0.3, rely=0.4, anchor="center")

txt2 = tk.Entry(window, font=('comic', 12), bg="#ffffff", fg="black", bd=2, relief="solid")
txt2.place(relx=0.51, rely=0.4, anchor="center", width=200)

clear_btn2 = tk.Button(window, text="Clear", command=clear2, bg="#DC3545", fg="white", font=('comic', 10))
clear_btn2.place(relx=0.7, rely=0.4, anchor="center")

role_var = tk.StringVar(value="User")
tk.Label(window, text="Select Role:", bg="#f0f0f0", font=("comic", 12)).place(relx=0.3, rely=0.5, anchor="center")
tk.Radiobutton(window, text="User", variable=role_var, value="User", bg="#f0f0f0", font=("comic", 10)).place(relx=0.45, rely=0.5, anchor="center")
tk.Radiobutton(window, text="Admin", variable=role_var, value="Admin", bg="#f0f0f0", font=("comic", 10)).place(relx=0.55, rely=0.5, anchor="center")

takeImg = tk.Button(window, text="Save Profile", command=TakeImages, bg="#007BFF", fg="white", font=('comic', 12))
takeImg.place(relx=0.5, rely=0.6, anchor="center", width=120)

window.mainloop()