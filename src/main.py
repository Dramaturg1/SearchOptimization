from PySide6.QtWidgets import QApplication, QVBoxLayout
import pyqtgraph.opengl as gl
from src.core.CustomLoader import CustomLoader
from core.plotter import generate_surface
from src.core.surfaces import surface_data
from src.methods.gradient_descent import GradientDescentMethod
import numpy as np
import sys
import os

surface_item = None
current_func = None
current_zmin = None
current_zmax = None
gd_method = None

def update_surface():
    global surface_item, current_func, current_zmin, current_zmax, gd_method

    try:
        xmin = float(window.lineEdit.text())
        xmax = float(window.lineEdit_2.text())
        ymin = float(window.lineEdit_3.text())
        ymax = float(window.lineEdit_4.text())
        npoints = int(window.lineEdit_5.text())
    except:
        window.textEdit.append("Ошибка")
        return

    name = window.comboBox.currentText()
    if name not in surface_data:
        return

    data = surface_data[name]
    func = data["func"]

    surface, Z_raw, Z_vis, zmin, zmax = generate_surface(
        func,
        xmin, xmax,
        ymin, ymax,
        npoints
    )

    current_zmin = zmin
    current_zmax = zmax

    if surface_item:
        view.removeItem(surface_item)

    surface.translate(0,0, -0.1)
    view.addItem(surface)
    surface_item = surface

    current_func = func
    current_zmin = zmin
    current_zmax = zmax

    if gd_method:
        gd_method.set_function(current_func, current_zmin, current_zmax)

    reset_view()

def reset_view():
    view.setCameraPosition(
        distance=30,
        elevation=30,
        azimuth=45
    )

def on_function_changed():
    name = window.comboBox.currentText()
    if name not in surface_data:
        return

    data = surface_data[name]
    window.lineEdit.setText(str(data["xmin"]))
    window.lineEdit_2.setText(str(data["xmax"]))
    window.lineEdit_3.setText(str(data["ymin"]))
    window.lineEdit_4.setText(str(data["ymax"]))
    window.lineEdit_5.setText(str(data["points"]))

def gradient_descent():
    global gd_method

    if current_func is None:
        window.textEdit.append("Сначала построй поверхность")
        return

    try:
        eps_grad = float(window.lineEdit_8.text())
        max_iter = int(window.lineEdit_9.text())
    except:
        window.textEdit.append("Ошибка параметров ГС")
        return

    if gd_method is None:
        gd_method = GradientDescentMethod(view, current_func, current_zmin, current_zmax, point_item, window)
    else:
        gd_method.reset()
        gd_method.set_function(current_func, current_zmin, current_zmax)

    try:
        x_start = float(window.lineEdit_6.text())
        y_start = float(window.lineEdit_7.text())
        start_points = [(x_start, y_start)]
    except ValueError:
        xmin = float(window.lineEdit.text())
        xmax = float(window.lineEdit_2.text())
        ymin = float(window.lineEdit_3.text())
        ymax = float(window.lineEdit_4.text())
        N = 100
        start_points = [
            (np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)) for _ in range(N)
        ]

    gd_method.run_multiple(start_points, eps_grad, max_iter)

def stop_gd():
    global gd_method
    if gd_method:
        gd_method.stop()

def reset_gd():
    global gd_method
    if gd_method:
        gd_method.reset()

app = QApplication.instance()
if app is None:
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

grid = gl.GLGridItem()
grid.setSize(10, 10)
grid.setSpacing(1, 1)
grid.translate(0, 0, -1)
view.addItem(grid)

axis = gl.GLAxisItem()
axis.setSize(5,5,5)
view.addItem(axis)

point_item = gl.GLScatterPlotItem(
    size=15,
    color=(1, 0, 0, 1)
)
point_item.setGLOptions('opaque')
view.addItem(point_item)

window.pushButton.clicked.connect(update_surface)
window.comboBox.currentTextChanged.connect(on_function_changed)
window.pushButton_2.clicked.connect(gradient_descent)
window.pushButton_4.clicked.connect(stop_gd)
window.pushButton_5.clicked.connect(reset_gd)

window.show()
sys.exit(app.exec())