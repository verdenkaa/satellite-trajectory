from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QGraphicsScene
import sys
from main import Earth, Satellite
import matplotlib.pyplot as plt
import numpy as np
import PIL
from scipy.integrate import odeint
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class Menu(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('main2.ui', self)
        self.start.clicked.connect(self.count)

    def count(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect((1, 1, 1))

        #ax.patch.set_facecolor('black')

        #plt.axis('off')
        #plt.grid(b=None)

        earth = Earth()
        latitude, longitude = 0, 0
        satellite = Satellite(latitude, longitude)

        earth.create()
        satellite.create()

        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Menu()
    form.show()
    sys.exit(app.exec())