"""
添加标注
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y = 2 * x + 1

plt.figure(num=1, figsize=(8, 5))
plt.plot(x, y)

# gca=get current axis
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 代替
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# 设置原点
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

x0 = 1
y0 = 2 * x0 + 1
plt.scatter(x0, y0, s=50, color='b')
# 画线
plt.plot([x0, x0], [y0, 0], 'k--', lw=1.0)
# method1 注释
plt.annotate(r'$2x+1=3$', xy=(x0, y0), xycoords='data',xytext=(+30,-30),
             textcoords='offset points',fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
# method2
plt.text(-3,3,r'$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',fontdict={'size':16,'color':'red'})
plt.show()
