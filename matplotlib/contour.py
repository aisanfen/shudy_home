# 等高线
import matplotlib.pyplot as plt
import numpy as np
def f(x,y):
    # the height functiom
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n=256
x=np.linspace(-3,3,n)
y=np.linspace(-3,3,n)
# 绑定网格值
X,Y=np.meshgrid(x,y)
# use plt.contourf to filling contour
# X,Y and value for (X,Y)point
# cmap 表示颜色的图谱
plt.contourf(X,Y,f(X,Y),8,alpha=0.8,cmap=plt.cm.hot)
# 画线
# 8表示把等高线分成几个区域
C=plt.contour(X,Y,f(X,Y),8,colors='black',linewidth=0.5)
# adding label
plt.clabel(C,inline=True,fonsize=10)
plt.xticks(())
plt.yticks(())
plt.show()