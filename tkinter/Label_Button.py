import tkinter as tk

window=tk.Tk()
window.title('my window')
window.geometry('300x200')           #尺寸
var=tk.StringVar()

l=tk.Label(window,textvariable=var,bg='blue',font=('Arial',12),width=25,height=5)

# 放置label
l.pack()
on_hit=False

def hit_me():
    global on_hit
    if on_hit==False:
        on_hit=True
        var.set('兔兔哥哥打我了')
    else:
        on_hit=False
        var.set('')
b=tk.Button(window,text='哭哭',width=25,height=2,command=hit_me)
b.pack()

# 相当于一个while循环
window.mainloop()