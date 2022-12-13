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
    r = np.array([0 - x, 0 - y, 0 - z])
    mr = np.linalg.norm(r) ** 3
    ax = G * (M_1 * (0 - x) / mr + M_2 * (0 - x) / mr)
    ay = G * (M_1 * (0 - y) / mr + M_2 * (0 - y) / mr)
    az = G * (M_1 * (0 - z) / mr + M_2 * (0 - z) / mr)
    return np.array([vx, vy, vz, ax, ay, az])


class Earth:
    def __init__(self, poligons=200):
        self.poligons = 100
        self.R = 6371
        self.R_atm = 6489
        self.mass = 5 * (10 ** 16)
        self.g = 9.8

    def create(self):
        u = np.linspace(-np.pi, np.pi, self.poligons)
        v = np.linspace(0, np.pi, self.poligons)

        x = self.R * np.outer(np.cos(u), np.sin(v))
        y = self.R * np.outer(np.sin(u), np.sin(v))
        z = self.R * np.outer(np.ones(np.size(u)), np.cos(v))

        im = PIL.Image.open('min_earth.png')
        im = np.array(im.resize([self.poligons, self.poligons])) / 255

        #ax.scatter(0, 0, 0, color="blue")
        ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=im, antialiased=True, shade=False)
        ax.scatter(15000, 15000, 15000, alpha=0)
        ax.scatter(-15000, 15000, 15000, alpha=0)
        ax.scatter(15000, -15000, 15000, alpha=0)
        ax.scatter(15000, 15000, -15000, alpha=0)




class Satellite:
    def __init__(self, latitude, longitude):
        self.hight = 1000

        x = earth.R + self.hight
        y = 0
        z = 0

        self.x, self.z = rotation(x, z, latitude)
        self.x, self.y = rotation(self.x, y, longitude)
        self.x, self.y, self.z = round(self.x), round(self.y), round(self.z)

        self.velocity = 8000
        self.mass = 100

        '''if -5 <= latitude <= 5:
            self.vx = 0
            self.vy = 0
            self.vz = 22
        else:
            self.vz = 0
            if longitude < 90:'''

        self.vx = 0
        self.vy = 0
        self.vz = 23




    def create(self):
        #ax.scatter(self.x, self.y, self.z, color="red")
        ts = np.linspace(0, 5000, 1000)
        x, z = rotation(self.x, self.z, 180 - latitude)
        x, y = rotation(x, self.y, 180 - longitude)
        state0 = np.array([x, y, z, self.vx, self.vy, self.vz])

        sol = odeint(state, state0, ts)
        xyz = min(sol, key=lambda x: math.sqrt((x[0] - self.x) ** 2 + (x[1] - self.y) ** 2 + (x[2] - self.z) ** 2))

        ax.scatter(self.x, self.y, self.z, color="red")

        ax.scatter(xyz[0], xyz[1], xyz[2], color="black")
        ax.plot(sol[:,0], sol[:,1], sol[:,2], 'g', label='Trajectory', linewidth=2.0)

        trajectory = [i[0:3] for i in sol]
        velocity = [i[3:6] for i in sol]

        trajectory_corrected = np.zeros(np.shape(trajectory))

        kinetic_enegry = []
        potential_enegry = []
        total_energy = []

        for i in sol:
            v = math.sqrt(i[3] ** 2 + i[4] ** 2 + i[5] ** 2)
            h = math.sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2)
            kinetic_enegry.append(satellite.mass * (v ** 2) / 2)
            potential_enegry.append(satellite.mass * 9.8 * h)
            total_energy.append(kinetic_enegry[-1] + potential_enegry[-1])


        ax2.plot(ts, kinetic_enegry, 'r', label="kinetic")
        ax2.plot(ts, potential_enegry, 'b', label="potential")
        ax2.plot(ts, total_energy, 'k', label="total")
        ax2.legend()
        #ax2.set_title(['change in total energy: ' + S.color_orbit[j] + ' orbit', total_energy[-1] - total_energy[0]])

        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')

        ax2.set_xlabel('Time')
        ax2.set_ylabel('Joule')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
ax2.legend()
ax.set_box_aspect((1, 1, 1))
ax.margins(8000, 8000, 8000)
ax.autoscale(enable=False, tight=True)

ax2.autoscale(enable=True, tight=False)

#ax.patch.set_facecolor('black')

#plt.axis('off')
#plt.grid(b="visible")


G = 6.6743015 * (10**(-11))

earth = Earth()
latitude, longitude = 90, 90
satellite = Satellite(latitude, longitude)

earth.create()
satellite.create()

plt.show()