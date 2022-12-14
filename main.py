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
    M_1 = 5 * (10 ** 16)
    M_2 = 100
    G = 6.6743015 * (10 ** (-11))
    x, y, z, vx, vy, vz = s
    r = np.array([0 - x, 0 - y, 0 - z])
    mr = np.linalg.norm(r) ** 3
    ax = G * (M_1 * (0 - x) / mr + M_2 * (0 - x) / mr)
    ay = G * (M_1 * (0 - y) / mr + M_2 * (0 - y) / mr)
    az = G * (M_1 * (0 - z) / mr + M_2 * (0 - z) / mr)
    return np.array([vx, vy, vz, ax, ay, az])


class Earth:
    def __init__(self, poligons=200):
        self.R = 6371
        self.R_atm = 6489
        self.mass = 5 * (10 ** 16)
        self.g = 9.8


    def create(self, ax, polygons):
        polygons *= 100
        u = np.linspace(-np.pi, np.pi, polygons)
        v = np.linspace(0, np.pi, polygons)

        x = self.R * np.outer(np.cos(u), np.sin(v))
        y = self.R * np.outer(np.sin(u), np.sin(v))
        z = self.R * np.outer(np.ones(np.size(u)), np.cos(v))

        im = PIL.Image.open('min_earth.png')
        im = np.array(im.resize([polygons, polygons])) / 255

        ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=im, antialiased=True, shade=False)

        return ax




class Satellite:
    def __init__(self, longitude, latitude):

        self.velocity = 8000
        self.mass = 100


        self.vx = 0
        self.vy = 0
        self.vz = 25




    def create(self, ax, ax2, longitude, latitude, hight, speed):

        x = 6371 + hight
        y = 0
        z = 0

        self.x, self.z = rotation(x, z, longitude)
        self.x, self.y = rotation(self.x, y, latitude)
        self.x, self.y, self.z = round(self.x), round(self.y), round(self.z)
        #ax.scatter(self.x, self.y, self.z, color="black")

        G = 6.6743015 * (10 ** (-11))

        ts = np.linspace(0, 10000, 1000)
        ts2 = np.linspace(0, 30000, 1000)
        z = 0
        x, y = rotation(x, y, latitude)

        state0 = np.array([x, y, z, 0, 0, speed])

        sol = odeint(state, state0, ts)


        xyz = min(sol, key=lambda x: math.sqrt((x[0] - self.x) ** 2 + (x[1] - self.y) ** 2 + (x[2] - self.z) ** 2))

        ax.scatter(xyz[0], xyz[1], xyz[2], color="red")
        ax.plot(sol[:,0], sol[:,1], sol[:,2], 'g', label='Trajectory', linewidth=2.0, color="green")
        #ax.scatter(sol[0][0], sol[0][1],)

        R = min(sol, key=lambda x: math.sqrt((x[0]) ** 2 + (x[1]) ** 2 + (x[2]) ** 2))
        R = math.sqrt((R[0]) ** 2 + (R[1]) ** 2 + (R[2]) ** 2)
        R_t = R < (6371 + 100)
        print("Атмосфера", R_t)

        R = min(sol, key=lambda x: math.sqrt((x[0]) ** 2 + (x[1]) ** 2 + (x[2]) ** 2))
        R = math.sqrt((R[0]) ** 2 + (R[1]) ** 2 + (R[2]) ** 2)
        crash = R < (6371)
        print("Падение", crash)

        line = sum([1 for i in sol if (i[0] - sol[0][0] < 100 and i[1] - sol[0][1] < 100 and i[2] - sol[0][2] < 100)])

        if line > 2:
            outgo = False
        else:
            outgo = True

        print("Out", outgo)

        kinetic_enegry = []
        potential_enegry = []
        total_energy = []

        for i in sol:
            v = math.sqrt(i[3] ** 2 + i[4] ** 2 + i[5] ** 2)
            h = math.sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2)
            kinetic_enegry.append(0.5 * self.mass * (v ** 2))
            potential_enegry.append(-1 * G * 5 * (10 ** 16) * self.mass / h)
            total_energy.append(kinetic_enegry[-1] + potential_enegry[-1])


        state0 = np.array([x, y, z, 0, 0, speed + abs(30 - speed)])

        sol = odeint(state, state0, ts2)
        ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], 'g', label='Trajectory', linewidth=2.0, color="orange")





        ax2.plot(ts, kinetic_enegry, 'r', label="kinetic")
        ax2.plot(ts, potential_enegry, 'b', label="potential")
        ax2.plot(ts, total_energy, 'k', label="total")
        ax2.legend()

        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')

        ax2.set_xlabel('Time')
        ax2.set_ylabel('Joule')

        return ax, ax2, outgo, R_t, crash
