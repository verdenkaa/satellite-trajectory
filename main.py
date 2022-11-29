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

'''def move(s, t):
    x, y, z, ax, ay, az = s

    R = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    F_t = G * (earth.mass * satellite.mass) / R ** 2
    F_s = satellite.mass * satellite.velocity

    V_e = np.array([0 - x, 0 - y, 0 - z])
    V_e = V_e / (V_e**2).sum()**0.5
    #print(V_e, "---", x, y, z)
    return np.array([ax, ay, az, ax, ay, az])'''

def move2(x, y, z, time):
    xl, yl, zl = [x,], [y,], [z,]
    F_s = satellite.mass * satellite.velocity
    for i in range(time):

        R = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        F_t = G * (earth.mass * satellite.mass) / R ** 2

        V_e = np.array([-x, -y, -z])  # вектор к земле
        V_e = V_e / (V_e ** 2).sum() ** 0.5

        V_north = np.array([-xl[-1], -yl[-1], 0])  # вектор к северу
        V_north = V_north / (V_north ** 2).sum() ** 0.5

        v_z, _ = rotation(-zl[-1], -xl[-1], 90)  # ортогональный вектор

        V_move = np.array([V_north[0], V_north[1], v_z])  # вектор движения итоговый

        x = x + V_move[0] * F_s
        y = y + V_move[1] * F_s
        z = z + V_move[2] * F_s
        xl.append(x)
        yl.append(y)
        zl.append(z)
        #print([xl[-1], yl[-1], zl[-1]])
    return [xl, yl, zl]


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
        self.hight = 2000 * 1000

        x = earth.R + self.hight
        y = 0
        z = 0


        self.x, self.z = rotation(x, z, latitude)
        self.x, self.y = rotation(self.x, y, longitude)

        self.position = np.array([self.x, self.y, self.z])
        self.velocity = 8000
        self.mass = 100



    def create(self):
        ax.scatter(self.x, self.y, self.z, color="red")
        ts = np.linspace(0, 1, 5)
        #sol = odeint(move,
                     #np.array([self.x, self.y, self.z, self.direction[0], self.direction[1], self.direction[2]]),
                     #ts)
        x, y, z = move2(self.x, self.y, self.z, 1000)

        ax.plot(x, y, z, linewidth=2.0, color="black")



        #ax.plot3D(X_Sat, Y_Sat, Z_Sat, 'black')
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

G = 6.6743015 * (10**(-11))
mu = 3.986004418E+05  # Earth's gravitational parameter

earth = Earth()
satellite = Satellite(0, 0)

earth.create()
satellite.create()

plt.show()