import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
# 添加3D网格图
ax=Axes3D(fig)
X=np.arange(-4,4,0.25)
Y=np.arange(-4,4,0.25)
# 讲XY放在里面
X,Y=np.meshgrid(X,Y)
R=np.sqrt(X**2+Y**2)
# height value
Z=np.sin(R)

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
# 表示等高线从Z轴压下去
ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
ax.set_zlim(-2,2)
plt.show()