from PySide6.QtWidgets import QApplication, QVBoxLayout
from PySide6.QtCore import QTimer
import pyqtgraph.opengl as gl
from src.core.CustomLoader import CustomLoader
from core.plotter import generate_surface
from src.core.surfaces import surface_data
from src.core.optimization_visualizer import OptimizationVisualizer
from src.methods.gradient import SteepestDescent
import numpy as np
import sys
import os

# Глобальные переменные
surface_item = None
current_func = None
current_grad = None
current_zmin = None
current_zmax = None
visualizer = None  # Будет хранить экземпляр OptimizationVisualizer
current_sd = None
timer = QTimer()
gd_running = False


def real_z_to_vis(z):
    """Конвертирует реальный Z в визуальный Z (0-10)"""
    if current_zmax == current_zmin:
        return current_zmax
    return (z - current_zmin) / (current_zmax - current_zmin) * 10


def update_surface():
    global surface_item, current_func, current_zmin, current_zmax, visualizer

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

    # Создаем градиент
    def grad_func(x):
        h = 1e-6
        dfdx = (func(x[0] + h, x[1]) - func(x[0] - h, x[1])) / (2 * h)
        dfdy = (func(x[0], x[1] + h) - func(x[0], x[1] - h)) / (2 * h)
        return np.array([dfdx, dfdy])

    current_grad = grad_func

    surface, Z_raw, Z_vis, zmin, zmax = generate_surface(
        func,
        xmin, xmax,
        ymin, ymax,
        npoints
    )

    current_zmin = zmin
    current_zmax = zmax
    current_func = func

    if surface_item:
        view.removeItem(surface_item)

    surface.translate(0, 0, -2)
    view.addItem(surface)
    surface_item = surface

    # Очищаем визуализатор
    if visualizer:
        visualizer.clear()

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


def start_optimization():
    global current_sd, visualizer, gd_running

    if current_func is None:
        print("Сначала построй поверхность")
        return

    try:
        x0 = float(window.lineEdit_6.text())
        y0 = float(window.lineEdit_7.text())
        eps = float(window.lineEdit_8.text())
        M = int(window.lineEdit_9.text())
    except ValueError:
        print("Ошибка параметров ГС")
        return

    # Создаем визуализатор если его нет
    if visualizer is None:
        visualizer = OptimizationVisualizer(view)

    # Создаем оптимизатор
    current_sd = SteepestDescent(
        current_func,
        current_grad,
        [x0, y0],
        eps=eps,
        M=M
    )

    visualizer.clear()
    gd_running = True

    # Добавляем начальную точку
    z0 = current_func(x0, y0)
    start_point = np.array([x0, y0, real_z_to_vis(z0)])
    visualizer.add_point(start_point, is_current=True)

    print(f"\n=== Запуск оптимизации ===")
    print(f"Начальная точка: [{x0}, {y0}]")
    print(f"Параметры: eps={eps}, M={M}")

    timer.start(500)


def step_optimization():
    global current_sd, visualizer, gd_running

    if not current_sd or not gd_running:
        return

    x, done, message = current_sd.step()

    # Конвертируем в визуальные координаты
    z = current_func(x[0], x[1])
    disp_point = np.array([x[0], x[1], real_z_to_vis(z)])

    if visualizer:
        visualizer.add_point(disp_point, is_current=True)

        # Добавляем стрелку от предыдущей точки
        if len(visualizer.points) >= 2:
            prev_point = visualizer.points[-2]
            visualizer.add_arrow(prev_point, disp_point)

    # Выводим каждые 5 итераций
    if current_sd.k % 5 == 0 or done:
        print(f"Итерация {current_sd.k}: точка [{x[0]:.6f}, {x[1]:.6f}], f={z:.6f}")

    if done:
        stop_optimization()
        print(f"✅ {message}")


def stop_optimization():
    global gd_running
    gd_running = False
    timer.stop()


def reset_optimization():
    global gd_running, current_sd, visualizer

    gd_running = False
    timer.stop()

    if visualizer:
        visualizer.clear()

    if current_sd:
        current_sd.reset()

    print("🔄 Сброс оптимизации")


# Инициализация приложения
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

# Загрузка UI
loader = CustomLoader()
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_path = os.path.join(current_dir, "ui", "main.ui")
window = loader.load(ui_path)

# Настройка 3D вида
view = gl.GLViewWidget(parent=window.widget)
layout = window.widget.layout()
if layout is None:
    layout = QVBoxLayout(window.widget)
    window.widget.setLayout(layout)
layout.addWidget(view)

# Сетка
grid = gl.GLGridItem()
grid.setSize(10, 10)
grid.setSpacing(1, 1)
grid.translate(0, 0, -1)
view.addItem(grid)

# Оси
axis = gl.GLAxisItem()
axis.setSize(5, 5, 5)
view.addItem(axis)

# Создаем визуализатор
visualizer = OptimizationVisualizer(view)

# Подключаем кнопки
window.pushButton.clicked.connect(update_surface)
window.comboBox.currentTextChanged.connect(on_function_changed)
window.pushButton_2.clicked.connect(start_optimization)  # Start
window.pushButton_3.clicked.connect(step_optimization)  # Step
window.pushButton_4.clicked.connect(stop_optimization)  # Stop
window.pushButton_5.clicked.connect(reset_optimization)  # Reset

# Таймер для автоматического шага
timer.timeout.connect(step_optimization)

window.show()
sys.exit(app.exec())