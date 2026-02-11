from PySide6.QtWidgets import QApplication, QVBoxLayout
import pyqtgraph.opengl as gl
from src.core.CustomLoader import CustomLoader
from core.plotter import generate_surface
from core.surfaces import surface_functions
import sys
import os

from src.core.surfaces import surface_functions

app = QApplication(sys.argv)

loader = CustomLoader()
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_path = os.path.join(current_dir, "ui", "main.ui")
window = loader.load(ui_path)

view = gl.GLViewWidget(parent=window.widget)
layout = window.widget.layout()
if layout is None:
    layout = QVBoxLayout(window.widget)
    window.widget.setLayout(layout)
layout.addWidget(view)
view.setCameraPosition(distance=30, elevation=30, azimuth=45)

grid = gl.GLGridItem()
grid.setSize(10, 10)
grid.setSpacing(1, 1)
view.addItem(grid)
surface_item = None

def update_surface():
    global surface_item
    try:
        xmin = float(window.lineEdit.text())
        xmax = float(window.lineEdit_2.text())
        ymin = float(window.lineEdit_3.text())
        ymax = float(window.lineEdit_4.text())
        npoints = int(window.lineEdit_5.text())
        zscale = 0
    except ValueError:
        print("Ошибка: неверные параметры")
        return
    func_name = window.comboBox.currentText()
    if func_name not in surface_functions:
        print("нету")
        return
    func = surface_functions[func_name]
    surface, Z = generate_surface(func, xmin, xmax, ymin, ymax, npoints)
    if surface_item:
        view.removeItem(surface_item)
    if zscale == 0:
        zscale = 10 / Z.max()
    surface.scale(1, 1, zscale)
    view.addItem(surface)
    surface_item = surface

def reset_view():
    view.setCameraPosition(distance=30, elevation=30, azimuth=30)
window.pushButton.clicked.connect(update_surface)

window.show()
sys.exit(app.exec())