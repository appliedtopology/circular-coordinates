import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets,QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import time
import sys

class ScrollableWindow(QtWidgets.QMainWindow):

    """
        scrollable window for plotting
        adapted from "https://stackoverflow.com/a/42624276"
     
            

    """

    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication.instance()
        if self.qapp is None:
            self.qapp = QtWidgets.QApplication(sys.argv)

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)
        self.show()
        self.qapp.exec_() 

    def closeEvent(self, event):
            plt.close(self.fig)
            event.accept() 
        
