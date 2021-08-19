# Import required Libraries
import tkinter as tk
from PIL import Image, ImageTk
import cv2

# Create an instance of TKinter Window or frame
win = tk.Tk()

# Set the size of the window
win.geometry("700x480")

# Create a Label to capture the Video frames
label = tk.Label(win)
label.grid(row=0, column=0)
cap = cv2.VideoCapture(0)

width  = cap.get(3)
height = cap.get(4)

print(width, height)
label.place(x=0, y=0)

# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   img = Image.fromarray(cv2image)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   # Repeat after an interval to capture continiously
   label.after(20, show_frames)

def clicked(event):
    print("CLICK")

button = tk.Button(
    text="OK",
    width=4,
    height=1,
    bg="blue",
    fg="yellow",
)
button.place(x=650, y=10)
button.bind("<Button-1>", clicked)

show_frames()
win.mainloop()