import subprocess
from threading import Thread
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox as mess
import cv2, os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from tkcalendar import Calendar, DateEntry
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

def update_attendance_table():
    tv.delete(*tv.get_children())
    date = datetime.datetime.now().strftime('%d-%m-%Y')
    file_path = f"Attendance/Attendance_{date}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            tv.insert("", "end", text=row['ID'], values=(row['Name'], row['Date'], row['Time']))

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) 
                if f.endswith('.jpg') or f.endswith('.png')]
    faces = []
    Ids = []
    
    for imagePath in imagePaths:
        try:
            ID = int(os.path.split(imagePath)[-1].split(".")[1])
            pilImage = Image.open(imagePath).convert('RGB')
            
            face = mtcnn(pilImage)
            if face is not None:
                if face.dim() == 4:  # [1, 3, 160, 160]
                    face = face.squeeze(0)  # [3, 160, 160]
                elif face.dim() == 5:  # [1, 1, 3, 160, 160]
                    face = face.squeeze(0).squeeze(0)  # [3, 160, 160]
                
                embedding = resnet(face.unsqueeze(0)).detach().cpu().numpy()
                faces.append(embedding)
                Ids.append(ID)
                print(f"Processed: {os.path.basename(imagePath)}")
            else:
                print(f"No face detected in: {os.path.basename(imagePath)}")
                
        except Exception as e:
            print(f"Error processing {os.path.basename(imagePath)}: {str(e)}")
            continue
    
    print(f"Total faces processed: {len(faces)}")
    return faces, Ids

def TrainImages():
    assure_path_exists("TrainingImageLabel/")
    try:
        faces, Ids = getImagesAndLabels("TrainingImage/")
        
        if not faces:
            mess.showerror("Error", "No faces found for training! Please add images first.")
            return
        
        faces_array = np.array(faces).squeeze()
        Ids_array = np.array(Ids)
        
        np.savez("TrainingImageLabel/Trainer.npz", faces=faces_array, Ids=Ids_array)
    except Exception as e:
        mess.showerror("Error", f"Failed to train model: {str(e)}")

def TrackImages(): 
    TrainImages
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")

    try:
        data = np.load("TrainingImageLabel/Trainer.npz")
        known_faces = data['faces']
        known_ids = data['Ids']
        print(f"Loaded {len(known_ids)} trained face embeddings")
    except Exception as e:
        mess.showerror("Error", f"Failed to load trained model: {str(e)}\nPlease train the model first!")
        return

    try:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        df['SERIAL NO.'] = df['SERIAL NO.'].astype(int)
        print("Loaded student details with shape:", df.shape)
    except Exception as e:
        mess.showerror("Error", f"Failed to load student details: {str(e)}")
        return

    date = datetime.datetime.now().strftime('%d-%m-%Y')
    attendance_file = f"Attendance/Attendance_{date}.csv"
    marked_ids = set()

    if os.path.exists(attendance_file):
        try:
            today_df = pd.read_csv(attendance_file)
            marked_ids = set(today_df['ID'].astype(str))
            print(f"Found {len(marked_ids)} already marked attendances")
        except Exception as e:
            print(f"Warning: Could not read attendance file - {str(e)}")

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cam.isOpened():
        mess.showerror("Error", "Could not open video capture")
        return

    last_detection_time = time.time()
    detection_interval = 0.5 
    recognition_state = {
        'consecutive_matches': 0,
        'last_id': None,
        'last_name': "",
        'marked': False,
        'current_box': None,
        'display_text': "Looking for faces...",
        'last_embedding': None,
        'spoofing_detected': False,
        'admin_interface_shown': False,
        'spoofing_counter': 0,  
        'real_frames_count': 0,  
        'last_spoofing_check_time': 0,
        'spoofing_check_interval': 2  
    }

    def close_camera():
        if cam.isOpened():
            cam.release()
        cv2.destroyAllWindows()

    def handle_admin(current_name):
        """Handle admin-specific actions"""
        if not recognition_state['admin_interface_shown']:
            recognition_state['admin_interface_shown'] = True
            response = mess.askyesno("Admin Detected", 
                                   f"Attendance marked for {current_name}\nOpen admin interface?")
            close_camera()
            if response:
                window.after(100, admin_options)
            return True
        return False

    while True:
        if recognition_state['spoofing_detected']:
            close_camera()
            break

        ret, frame = cam.read()
        if not ret:
            print("Warning: Could not read frame from camera")
            continue

        display_frame = frame.copy()
        current_time = time.time()

        if recognition_state['current_box'] is not None:
            box_color = (0, 0, 255) if recognition_state['spoofing_detected'] else (0, 255, 0)
            x1, y1, x2, y2 = recognition_state['current_box']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(display_frame, recognition_state['display_text'], 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        if current_time - last_detection_time > detection_interval and not recognition_state['spoofing_detected']:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(frame_rgb)

                if boxes is not None and len(boxes) > 0:
                    main_box = max(boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
                    x1, y1, x2, y2 = [int(coord) for coord in main_box]
                    recognition_state['current_box'] = (x1, y1, x2, y2)

                    if x2 > x1 and y2 > y1:
                        face = frame_rgb[y1:y2, x1:x2]
                        if face.size > 0:
                            face_pil = Image.fromarray(face)

                            try:
                                face_tensor = mtcnn(face_pil)
                                if face_tensor is not None:
                                    while face_tensor.dim() > 4:
                                        face_tensor = face_tensor.squeeze(0)
                                    if face_tensor.dim() == 4:
                                        face_tensor = face_tensor.squeeze(0)

                                    face_tensor = face_tensor.to(device)
                                    current_embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()

                                    if (current_time - recognition_state['last_spoofing_check_time'] > 
                                        recognition_state['spoofing_check_interval']):

                                        if recognition_state['last_embedding'] is not None:
                                            embedding_diff = np.linalg.norm(
                                                recognition_state['last_embedding'] - current_embedding
                                            )
                                            print(f"Embedding difference: {embedding_diff:.4f}")

                                            if embedding_diff < 0.5:  
                                                recognition_state['spoofing_counter'] += 1
                                                recognition_state['real_frames_count'] = 0

                                                
                                                if recognition_state['spoofing_counter'] >= 5:
                                                    recognition_state['spoofing_detected'] = True
                                                    recognition_state['display_text'] = "SPOOFING DETECTED!"
                                                    mess.showerror("Security Alert", 
                                                                 "Spoofing attempt detected!\nSystem will now close.")
                                                    close_camera()
                                                    break
                                            else:
                                                
                                                recognition_state['spoofing_counter'] = 0
                                                recognition_state['real_frames_count'] += 1

                                                if recognition_state['real_frames_count'] >= 7:
                                                    recognition_state['spoofing_detected'] = False

                                        recognition_state['last_spoofing_check_time'] = current_time
                                        recognition_state['last_embedding'] = current_embedding

                                    min_distance = float('inf')
                                    best_match_idx = None

                                    for idx, known_face in enumerate(known_faces):
                                        distance = np.linalg.norm(known_face - current_embedding)
                                        if distance < min_distance:
                                            min_distance = distance
                                            best_match_idx = idx

                                    print(f"Recognition distance: {min_distance:.4f}")

                                    if min_distance < 0.7: 
                                        recognized_id = known_ids[best_match_idx]
                                        student_info = df[df['SERIAL NO.'] == recognized_id]

                                        if not student_info.empty:
                                            current_id = str(student_info['ID'].values[0])
                                            current_name = student_info['NAME'].values[0]
                                            role = student_info['ROLE'].values[0]

                                
                                            if current_id == recognition_state['last_id']:
                                                recognition_state['consecutive_matches'] += 1
                                            else:
                                                recognition_state['consecutive_matches'] = 1
                                                recognition_state['last_id'] = current_id
                                                recognition_state['last_name'] = current_name
                                                recognition_state['marked'] = False
                                                recognition_state['admin_interface_shown'] = False

                                            recognition_state['display_text'] = current_name

                                            if recognition_state['consecutive_matches'] >= 3:
                                                if current_id not in marked_ids:
                                                    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                                                    attendance = [current_id, current_name, date, timestamp]

                                                    try:
                                                        with open(attendance_file, 'a+', newline='') as csvFile:
                                                            writer = csv.writer(csvFile)
                                                            if os.stat(attendance_file).st_size == 0:
                                                                writer.writerow(['ID', 'Name', 'Date', 'Time'])
                                                            writer.writerow(attendance)

                                                        marked_ids.add(current_id)
                                                        update_attendance_table()
                                                        recognition_state['marked'] = True

                                                        if role == "Admin":
                                                            if handle_admin(current_name):
                                                                break
                                                        else:
                                                            mess.showinfo("Success", f"Attendance marked for {current_name}")
                                                    except Exception as e:
                                                        print(f"Error saving attendance: {str(e)}")
                                                else:
                                                    if role == "Admin":
                                                        if handle_admin(current_name):
                                                            break
                                                    else:
                                                        mess.showinfo("Info", f"Attendance already marked for {current_name}")
                            except Exception as e:
                                print(f"Error processing face: {str(e)}")
            except Exception as e:
                print(f"Error in face detection: {str(e)}")

            last_detection_time = current_time

        cv2.imshow('Attendance System', display_frame)
        key = cv2.waitKey(1)
        if key == ord('q') or recognition_state['spoofing_detected']:
            close_camera()
            break

    # Final cleanup
    close_camera()

def export_today_attendance():
    date = datetime.datetime.now().strftime('%d-%m-%Y')
    attendance_file = f"Attendance/Attendance_{date}.csv"
    
    if not os.path.exists(attendance_file):
        mess.showinfo("No Data", "No attendance records found for today")
        return
    
    try:
        # Ask where to save the file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"Today_Attendance_{date}.csv",
            title="Save Today's Attendance As"
        )
        
        if file_path: 
            import shutil
            shutil.copy(attendance_file, file_path)
            mess.showinfo("Success", f"Today's attendance exported to:\n{file_path}")
    except Exception as e:
        mess.showerror("Error", f"Failed to export attendance: {str(e)}")

def admin_options():
    admin_win = tk.Toplevel()
    admin_win.title("Admin Options")
    admin_win.geometry("400x300")
    
    try:
        bg_image = Image.open("img.png")
        bg_photo = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(admin_win, image=bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.image = bg_photo 
    except Exception as e:
        print(f"Error loading background image: {e}")
       
        admin_win.config(bg="#f0f0f0")
    
    button_style = {'font': ('comic', 12), 'bg': '#007BFF', 'fg': 'white', 'height': 2}
    
    content_frame = tk.Frame(admin_win, bg='white', bd=2, relief=tk.RAISED)
    content_frame.place(relx=0.5, rely=0.5, anchor='center', width=380, height=300)
    
    tk.Label(content_frame, text="Admin Panel", font=('comic', 16, 'bold'), bg='white').pack(pady=10)
    
    btn_frame = tk.Frame(content_frame, bg='white')
    btn_frame.pack(pady=20)
    
 
    tk.Button(btn_frame, text="Register New User", 
             command=lambda: subprocess.Popen(["python", "registration.py"]), **button_style).pack(pady=5, fill=tk.X)
    
    tk.Button(btn_frame, text="Generate Attendance Report", 
             command=generate_report_window, **button_style).pack(pady=5, fill=tk.X)
    
    tk.Button(btn_frame, text="Export Today's Attendance", 
             command=export_today_attendance, **button_style).pack(pady=5, fill=tk.X)
    
    admin_win.grab_set()
def generate_report_window():
    report_win = tk.Toplevel()
    report_win.title("Generate Report")
    report_win.geometry("600x500")
    
    bg_image = Image.open("img.png")
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(report_win, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)
    
    frame = tk.Frame(report_win, bg="white", bd=2, relief=tk.RAISED)
    frame.place(relx=0.5, rely=0.5, anchor="center", width=550, height=400)  # Increased size
    
    tk.Label(frame, text="Generate Attendance Report", bg="blueviolet", fg="white", 
            font=('comic', 15, 'bold')).pack(fill=tk.X, pady=10)
    
    date_frame = tk.Frame(frame, bg="white")
    date_frame.pack(pady=10)
    
    tk.Label(date_frame, text="From Date:", font=('comic', 12), bg="white").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    from_date = DateEntry(date_frame, font=('comic', 12), date_pattern='dd-mm-yyyy',
                         mindate=datetime.date(2020, 1, 1),
                         maxdate=datetime.date.today())
    from_date.grid(row=0, column=1, padx=5, pady=5)
    

    tk.Label(date_frame, text="To Date:", font=('comic', 12), bg="white").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    to_date = DateEntry(date_frame, font=('comic', 12), date_pattern='dd-mm-yyyy',
                       mindate=datetime.date(2020, 1, 1),
                       maxdate=datetime.date.today())
    to_date.grid(row=1, column=1, padx=5, pady=5)
    
    to_date.set_date(datetime.date.today())
    from_date.set_date(datetime.date.today() - datetime.timedelta(days=7))
    

    type_frame = tk.Frame(frame, bg="white")
    type_frame.pack(pady=10)
    
    report_type = tk.StringVar(value="daily")
    tk.Radiobutton(type_frame, text="Daily Summary", variable=report_type, value="daily", 
                  font=('comic', 11), bg="white").pack(side=tk.LEFT, padx=10)
    tk.Radiobutton(type_frame, text="Detailed Report", variable=report_type, value="detailed", 
                  font=('comic', 11), bg="white").pack(side=tk.LEFT, padx=10)
    
    def generate():
        try:
            start = from_date.get_date().strftime('%d-%m-%Y')
            end = to_date.get_date().strftime('%d-%m-%Y')
            r_type = report_type.get()
            
            start_date = datetime.datetime.strptime(start, "%d-%m-%Y")
            end_date = datetime.datetime.strptime(end, "%d-%m-%Y")
            
            all_dfs = []
            current = start_date
            while current <= end_date:
                date_str = current.strftime("%d-%m-%Y")
                file_path = f"Attendance/Attendance_{date_str}.csv"
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df['Date'] = date_str
                    all_dfs.append(df)
                current += datetime.timedelta(days=1)
            
            if not all_dfs:
                mess.showinfo("No Data", "No attendance records found for selected dates")
                return
            
            combined = pd.concat(all_dfs)
            
            if r_type == "daily":
                report = combined.groupby(['Date', 'Name']).size().unstack(fill_value=0)
            else:
                report = combined.sort_values(['Date', 'Time'])
           
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"Attendance_Report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv",
                title="Save Report As"
            )
            
            if file_path: 
                report.to_csv(file_path)
                mess.showinfo("Success", f"Report generated at:\n{file_path}")
                report_win.destroy()
            
        except Exception as e:
            mess.showerror("Error", f"Error generating report:\n{e}")
    
    tk.Button(frame, text="Generate", command=generate, bg="#007BFF", fg="white", 
             font=('comic', 12)).pack(pady=20, fill=tk.X, padx=50)
    
    report_win.mainloop()

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")

mont = {
    '01': 'January',
    '02': 'February',
    '03': 'March',
    '04': 'April',
    '05': 'May',
    '06': 'June',
    '07': 'July',
    '08': 'August',
    '09': 'September',
    '10': 'October',
    '11': 'November',
    '12': 'December'
}

window = tk.Tk()
window.geometry("1280x720")
window.resizable(True, False)
window.title("Attendance System")

bg_image = Image.open("img.png")
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(window, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

frame1 = tk.Frame(window)
frame1.place(relx=0.5, rely=0.17, relwidth=0.39, relheight=0.80, anchor='n')

message3 = tk.Label(window, text="Attendance System", fg="black", width=20, height=1, anchor='center', font=('comic', 29, ' bold '))
message3.place(relx=0.5, y=10, anchor='n')

frame3 = tk.Frame(window)
frame3.place(relx=0.53, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window)
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(frame4, text=day + "-" + mont[month] + "-" + year + "   ", fg="black", width=55, height=1, font=('comic', 20, ' bold '))
datef.pack(fill='both', expand=1)

clock = tk.Label(frame3, fg="black", width=55, height=1, font=('comic', 22, ' bold '))
clock.pack(fill='both', expand=1)
tick()

head1 = tk.Label(frame1, text="                       For Already Registered                       ", fg="white", bg="blueviolet", font=('comic', 17, ' bold '))
head1.place(x=0, y=0)

lbl3 = tk.Label(frame1, text="Attendance", width=20, fg="black", bg="#c79cff", height=1, font=('comic', 17, ' bold '))
lbl3.place(x=100, y=115)

tv = ttk.Treeview(frame1, height=13, columns=('name', 'date', 'time'))
tv.column('#0', width=82)
tv.column('name', width=130)
tv.column('date', width=133)
tv.column('time', width=133)
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
tv.heading('#0', text='ID')
tv.heading('name', text='NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')

scroll = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

trackImg = tk.Button(frame1, text="Take Attendance", command=TrackImages, fg="black", bg="#3ffc00", width=35, height=1, activebackground="white", font=('comic', 15, ' bold '))
trackImg.place(x=30, y=50)

quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, fg="black", bg="#eb4600", width=35, height=1, activebackground="white", font=('comic', 15, ' bold '))
quitWindow.place(x=30, y=450)

update_attendance_table()
window.mainloop()