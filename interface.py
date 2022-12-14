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
        uic.loadUi('main.ui', self)
        self.start.clicked.connect(self.count)

    def count(self):
        lo = self.longtitude.value()
        la = self.latitude.value()
        hight = self.hight.value()
        speed = self.speed.value()
        polygons = self.polygons.value()
        space = self.space.isChecked()

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        ax2.legend()
        ax.set_box_aspect((1, 1, 1))
        ax.margins(8000, 8000, 8000)
        ax.autoscale(enable=False, tight=True)

        if space:
            ax.patch.set_facecolor('black')
            ax.set_axis_off()


        earth = Earth()
        satellite = Satellite(lo, la)

        ax = earth.create(ax, polygons)
        ax, ax2 = satellite.create(ax, ax2, lo, la, hight, speed)

        plt.show()


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Menu()
    form.show()
    sys.excepthook = except_hook
    sys.exit(app.exec())