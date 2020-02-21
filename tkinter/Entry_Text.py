import tkinter as tk

window = tk.Tk()
window.title('my window')
window.geometry('200x200')  # 尺寸
e = tk.Entry(window, show=None)
e.pack()   # 放在窗口上


def insert_point():
    var = e.get()
    t.insert('insert', var)


def insert_end():
    var = e.get()
    t.insert('end', var)


# 插入到当前位置
b1 = tk.Button(
    window,
    text='insert point',
    width=15,
    height=2,
    command=insert_point)
b1.pack()
# 插入到最后
b2 = tk.Button(window, text='insert_end', command=insert_end)
b2.pack()
# 文本框
t = tk.Text(window, height=2)
t.pack()
# 相当于一个while循环
window.mainloop()
