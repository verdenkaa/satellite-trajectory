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
        self.mass = 5.9722 * 10 ** 24
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
        self.x = 6489
        self.y = 6489
        self.z = 6489
        self.position = [self.x, self.y, self.z]
        self.velocity = 100
        self.mass = 100
        self.hight = 200
        self.direction = [5, -3, -2]

    def create(self, i):
        r = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        F = G * (self.mass * earth.mass / r ** 2)
        x2 = self.x + self.velocity * self.direction[0]
        y2 = self.y + self.velocity * self.direction[1]
        z2 = self.z + self.velocity * self.direction[2]
        self.x = x2
        self.y = y2
        self.z = z2
        self.position = [self.x, self.y, self.z]
        ax.scatter(x2, y2, z2, color="red")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
G = 6.6743015 * 10**(-11)
print(G, "---------G")

earth = Earth()
satellite = Satellite()

earth.create()
#planet.create_atmosphere()

def animate(i):
    satellite.create(i)

ani = animation.FuncAnimation(fig, animate, interval=0)
plt.show()