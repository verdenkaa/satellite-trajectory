import matplotlib.pyplot as plt
import numpy as np
import PIL
from scipy.integrate import odeint
import math

def rotation(x, y, a):
    a = math.radians(a)
    x2 = x * round(math.cos(a), 5) - y * math.sin(a)
    y2 = x * math.sin(a) + y * round(math.cos(a), 5)
    return (x2, y2)

def state(s: np.ndarray, t: float) -> np.ndarray:
    M_1 = earth.mass
    M_2 = satellite.mass
    x, y, z, vx, vy, vz = s
    r1 = np.array([0 - x, 0 - y, 0 - z])
    mr1 = np.linalg.norm(r1) ** 3
    ax = G * (M_1 * (0 - x) / mr1 + M_2 * (0 - x) / mr1)
    ay = G * (M_1 * (0 - y) / mr1 + M_2 * (0 - y) / mr1)
    az = G * (M_1 * (0 - z) / mr1 + M_2 * (0 - z) / mr1)
    return np.array([vx, vy, vz, ax, ay, az])


class Earth:
    def __init__(self):
        self.poligons = 200
        self.R = 6371
        self.R_atm = 6489
        self.mass = 5.972 * (10 ** 24)
        self.mass = 100000
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


class Satellite:
    def __init__(self, latitude, longitude):
        self.hight = 2000

        x = earth.R + self.hight
        y = 0
        z = 0


        self.x, self.z = rotation(x, z, latitude)
        self.x, self.y = rotation(self.x, y, longitude)

        self.velocity = 8000
        self.mass = 100
        self.mass = 10

        self.vx = 0
        self.vy = 0
        self.vz = 3.5



    def create(self):
        ax.scatter(self.x, self.y, self.z, color="red")
        ts = np.linspace(0, 100000, 1000)
        state0 = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz])

        sol = odeint(state, state0, ts)

        ax.plot(sol[:,0], sol[:,1], sol[:,2], 'g', label='Trajectory', linewidth=2.0)



        #ax.plot3D(X_Sat, Y_Sat, Z_Sat, 'black')
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect((1, 1, 1))



G = 6.6743015 * (10**(-11))
G = 1
mu = 3.986004418E+05  # Earth's gravitational parameter

earth = Earth()
satellite = Satellite(0, 0)

earth.create()
satellite.create()

plt.show()