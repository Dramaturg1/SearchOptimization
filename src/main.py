from PySide6.QtWidgets import QApplication, QVBoxLayout
import pyqtgraph.opengl as gl
from src.core.CustomLoader import CustomLoader
from core.plotter import generate_surface
from src.core.surfaces import surface_data
import numpy as np
import sys
import os

surface_item = None
current_func = None
current_zmin = None
current_zmax = None
trajectory_items = []
trajectory_points = []

gd_running = False

def z_to_vis(z):
    return (z - current_zmin) / (current_zmax - current_zmin) * 10

def update_surface():

    global surface_item
    global current_func, current_zmin, current_zmax

    try:
        xmin = float(window.lineEdit.text())
        xmax = float(window.lineEdit_2.text())
        ymin = float(window.lineEdit_3.text())
        ymax = float(window.lineEdit_4.text())
        npoints = int(window.lineEdit_5.text())
    except:
        print("Ошибка параметров")
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

    surface.translate(0,0, -2)
    view.addItem(surface)
    surface_item = surface

    current_func = func
    current_zmin = zmin
    current_zmax = zmax

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


def real_z_to_vis(z):
    if current_zmax==current_zmin:
        return current_zmax
    return (z - current_zmin) / (current_zmax - current_zmin) * 10

def show_point(x, y):
    if current_func is None:
        return
    z = current_func(x, y)
    z_vis = real_z_to_vis(z)
    pos = np.array([[x, y, z_vis + 0.05]])
    point_item.setData(pos=pos)

def random_color():
    return (np.random.rand(), np.random.rand(), np.random.rand(), 1.0)

def gradient(f, x, y, h=1e-5):

    dx = (f(x+h, y) - f(x-h, y)) / (2*h)
    dy = (f(x, y+h) - f(x, y-h)) / (2*h)

    return dx, dy

def gradient_descent():
    global gd_running, trajectory_items

    if current_func is None:
        print("Сначала построй поверхность")
        return

    try:
        eps_grad = float(window.lineEdit_8.text())
        max_iter = int(window.lineEdit_9.text())
    except:
        print("Ошибка параметров ГС")
        return

    lr = 0.01
    eps_pos = 1e-5
    eps_f = 1e-6

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
            (np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)) for _ in range (N)]

    gd_running = True
    minima = []
    for x0, y0 in start_points:
        if not gd_running:
            break

        x, y = x0, y0
        f_prev = current_func(x, y)
        traj_points = []
        color = random_color()
        traj_item = gl.GLLinePlotItem(color=color, width=3)
        view.addItem(traj_item)
        trajectory_items.append(traj_item)

        for k in range(max_iter):
            if not gd_running:
                break

            dx, dy = gradient(current_func, x, y)
            grad_norm = np.sqrt(dx**2 + dy**2)
            if grad_norm < eps_grad:
                break

            x_new = x - lr * dx
            y_new = y - lr * dy
            f_new = current_func(x_new, y_new)

            if f_new > f_prev:
                lr *= 0.5
                continue

            if np.sqrt((x_new - x)**2 + (y_new - y)**2) < eps_pos and abs(f_new - f_prev) < eps_f:
                x, y = x_new, y_new
                break

            x, y = x_new, y_new
            f_prev = f_new

            show_point(x, y)

            pos = np.array([[x, y, real_z_to_vis(current_func(x, y))]])
            traj_points.append(pos)
            traj_item.setData(pos=np.array(traj_points))

            QApplication.processEvents()

        minima.append((x, y, current_func(x, y)))
        print(f"Старт ({x0:.2f},{y0:.2f}): минимум найден: x={x:.5f}, y={y:.5f}, f={current_func(x, y):.5f}")

    if minima:
        global_min = min(minima, key=lambda t: t[2])
        print(f"\nГлобальный минимум среди всех стартов: x={global_min[0]:.5f}, "
              f"y={global_min[1]:.5f}, f={global_min[2]:.5f}")

def stop_gd():
    global gd_running
    gd_running = False


def reset_gd():
    global gd_running, trajectory_items

    gd_running = False
    for item in trajectory_items:
        view.removeItem(item)
    trajectory_items = []

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


# Point
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