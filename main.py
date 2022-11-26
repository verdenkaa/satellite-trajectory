import matplotlib.pyplot as plt
import numpy as np
import PIL
import utm
import matplotlib.animation as animation
from vectors import Vector
import math

def rotation(x, y, a):
    a = math.radians(a)
    x2 = x * round(math.cos(a), 5) - y * math.sin(a)
    y2 = x * math.sin(a) + y * round(math.cos(a), 5)
    return (x2, y2)
class Earth:
    def __init__(self):
        self.poligons = 200
        self.R = 6371 * 1000
        self.R_atm = 6489
        self.mass = 5.972 * (10 ** 24)
        self.g = 9.8

    def create(self):
        u = np.linspace(-np.pi, np.pi, self.poligons)
        v = np.linspace(0, np.pi, self.poligons)

        x = self.R * np.outer(np.cos(u), np.sin(v))
        y = self.R * np.outer(np.sin(u), np.sin(v))
        z = self.R * np.outer(np.ones(np.size(u)), np.cos(v))

        im = PIL.Image.open('min_earth.png')
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
    def __init__(self, latitude, longitude):
        self.hight = 1000 * 1000

        x = earth.R + self.hight
        y = 0
        z = 0
        self.x = -earth.R - self.hight
        self.y = 0
        self.z = 0

        self.x, self.z = rotation(x, z, latitude)
        self.x, self.y = rotation(self.x, y, longitude)

        print(x, y, z, "coordsNON")
        print(self.x, self.y, self.z, "coords")
        self.position = [self.x, self.y, self.z]
        self.velocity = 1000
        self.mass = 100
        self.direction = np.array([1, 0, 0])

    def create(self):
        ax.scatter(self.x, self.y, self.z, color="red")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

G = 6.6743015 * (10**(-11))

earth = Earth()
satellite = Satellite(90, 90)

earth.create()
#planet.create_atmosphere()

satellite.create()

plt.show()