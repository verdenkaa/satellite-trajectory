import matplotlib as mtl
import matplotlib.pyplot as plt
import matplotlib.image as image
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection = '3d')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

#im = image.imread('D:/earth.jpg')
im = cv2.imread('D:/min_earth.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = im.astype('float32')/255

ax.plot_surface(x, y, z, facecolors=im, rstride=4, cstride=4, antialiased=True, shade=False)

#x = np.linspace(0, 2 * np.pi, 100)
#y = 1*np.cos(x)
#z = 1*np.sin(x)
#ax.plot(y*np.sin(np.pi)+np.sin(np.pi),
            #y*np.cos(np.pi)+np.cos(np.pi), z, color="r")

plt.show()