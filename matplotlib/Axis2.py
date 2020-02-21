import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-3,3,50)
y1=2*x+1
y2=x**2
plt.figure()
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
# 设置坐标轴
plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('i am x')
plt.ylabel('i am y')
new_ticks=np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3],
           [r'$really\ bad$',r'$bad$',r'$normal$',r'$good$',r'$really\ good$'])

# gca=get current axis
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 代替
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# 设置原点
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
plt.show()