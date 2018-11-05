#encoding:utf-8
from Tkinter import *
from PIL import Image,ImageTk
import use_opencv

gui_title = 'opencv'
image1_address = ''
image2_address = ''
root1 = Tk()
root1.title(gui_title)
root1.geometry("1200x800")

f1 = Frame(root1)
f1.pack(expand='yes')

f2 = Frame(root1)
f2.pack(expand='yes')

def funtion1():
    global image_1,label_1,image_2,label_2
    address = text_address.get()
    image_1 = ImageTk.PhotoImage(file=address)
    label_1.configure(image = image_1)
    use_opencv.opencvapi(address)
    image_2 = ImageTk.PhotoImage(file='2/'+ address)
    label_2.configure(image= image_2)

#输入框
var = StringVar()
text_address = Entry(f1, textvariable = var, width=80)
text_address.grid(row=0, column=0, rowspan=1, columnspan=3)

#确定按钮
button_s = Button(f1, text ='确定', font = ('Arial', 20), width=10, command=funtion1)
button_s.grid(row=0, column=3, rowspan=1, columnspan=1)


#图片１
label_1 = Label(f2,width=600,height=800)
label_1.grid(row=0, column=0, rowspan=2, columnspan=2)


# #图片２
label_2 = Label(f2, width=600,height=800)
label_2.grid(row=0, column=2, rowspan=2, columnspan=2)

root1.mainloop()
