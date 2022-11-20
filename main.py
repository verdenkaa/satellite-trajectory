import matplotlib.pyplot as plt
import numpy as np
import PIL
import matplotlib.animation as animation
from vectors import Vector
import math

class Earth:
    def __init__(self):
        self.poligons = 100
        self.R = 6371
        self.R_atm = 6489
        self.mass = 5.972 * (10 ** 24)
        print(self.mass, "M")
        self.g = 9.8

    def create(self):
        u = np.linspace(0, 2 * np.pi, self.poligons)
        v = np.linspace(0, np.pi, self.poligons)

        x = self.R * np.outer(np.cos(u), np.sin(v))
        y = self.R * np.outer(np.sin(u), np.sin(v))
        z = self.R * np.outer(np.ones(np.size(u)), np.cos(v))

        im = PIL.Image.open('min_earth.png')
        print(im.size)
        im = np.array(im.resize([self.poligons, self.poligons])) / 255

        ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=im, antialiased=True, shade=False)

    def create_atmosphere(self):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = self.R_atm * np.outer(np.cos(u), np.sin(v))
        y = self.R_atm * np.outer(np.sin(u), np.sin(v))
        z = self.R_atm * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, rstride=4, cstride=4, color="blue", antialiased=True, shade=False, alpha=0.1)


class Satellite:
    def __init__(self):
        self.hight = 0
        self.x = earth.R + self.hight
        self.y = earth.R + self.hight
        self.z = earth.R + self.hight
        print(self.x, self.y, self.z)
        self.position = [self.x, self.y, self.z]
        self.velocity = 10000
        self.mass = 1000
        self.direction = np.array([1, 0, 0])

    def create(self, i):
        r = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        print(r, "R")
        direction_earth = np.array([0 - self.x, 0 - self.y, 0 - self.z])
        direction_earth = direction_earth / np.linalg.norm(direction_earth)
        F = self.mass * earth.g #G * ((self.mass * earth.mass) / (r ** 2))
        #print(F)
        x2 = self.x + self.velocity * (direction_earth[0] + self.direction[0]) // 2
        y2 = self.y + self.velocity * (direction_earth[1] + self.direction[1]) // 2
        z2 = self.z + self.velocity * (direction_earth[2] + self.direction[2]) // 2
        self.x = x2
        self.y = y2
        self.z = z2
        self.position = [self.x, self.y, self.z]
        ax.scatter(x2, y2, z2, color="red")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

G = 6.6743015 * (10**(-11))
print(G, "---------G")

earth = Earth()
satellite = Satellite()

earth.create()
#planet.create_atmosphere()

def animate(i):
    satellite.create(i)

ani = animation.FuncAnimation(fig, animate, interval=0)
plt.show()