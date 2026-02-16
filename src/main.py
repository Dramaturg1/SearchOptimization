from PySide6.QtWidgets import QApplication, QVBoxLayout, QTextEdit
from PySide6.QtCore import QTimer
import pyqtgraph.opengl as gl
from src.core.CustomLoader import CustomLoader
from core.plotter import generate_surface
from src.core.surfaces import surface_data
from src.methods.gradient import SteepestDescent
import sys
import os
import numpy as np

# Глобальные переменные
surface_item = None
optimization_visualizer = None
current_func = None
current_grad = None
current_sd = None
timer = QTimer()
console_output = None

# Параметры для конвертации координат (как в plotter.py)
current_z_min = 0
current_z_max = 0


def update_surface():
    global surface_item, optimization_visualizer, current_func, current_grad
    global current_z_min, current_z_max

    try:
        xmin = float(window.lineEdit.text())
        xmax = float(window.lineEdit_2.text())
        ymin = float(window.lineEdit_3.text())
        ymax = float(window.lineEdit_4.text())
        npoints = int(window.lineEdit_5.text())
    except ValueError:
        print("Ошибка: неверные параметры")
        return

    func_name = window.comboBox.currentText()
    if func_name not in surface_data:
        print("нету")
        return

    # Сохраняем функцию
    current_func = surface_data[func_name]["func"]

    # Создаем градиент
    def grad_func(x):
        h = 1e-6
        dfdx = (current_func(x[0] + h, x[1]) - current_func(x[0] - h, x[1])) / (2 * h)
        dfdy = (current_func(x[0], x[1] + h) - current_func(x[0], x[1] - h)) / (2 * h)
        return np.array([dfdx, dfdy])

    current_grad = grad_func

    # Генерируем поверхность (она уже нормализована в plotter.py)
    func = surface_data[func_name]["func"]
    surface, Z = generate_surface(func, xmin, xmax, ymin, ymax, npoints)

    # Сохраняем параметры нормализации
    current_z_min = Z.min()
    current_z_max = Z.max()

    if surface_item:
        view.removeItem(surface_item)

    view.addItem(surface)
    surface_item = surface

    # Очищаем визуализатор
    if optimization_visualizer:
        optimization_visualizer.clear()

    # Обновляем информацию
    print_to_console(f"Функция загружена: {func_name}")
    print_to_console(f"Область: x∈[{xmin},{xmax}], y∈[{ymin},{ymax}]")
    print_to_console(f"Z range (нормализованный): [{current_z_min:.2f}, {current_z_max:.2f}]")


def to_display_coords(x, y, z):
    """Конвертирует координаты функции в координаты отображения"""
    z_range = current_z_max - current_z_min
    if z_range == 0:
        z_norm = 5.0
    else:
        z_norm = (z - current_z_min) / z_range * 10
    return np.array([float(x), float(y), z_norm])


def setup_console():
    """Настраиваем консольный вывод"""
    global console_output

    console_output = QTextEdit()
    console_output.setReadOnly(True)
    console_output.setFontFamily("Courier New")
    console_output.setMaximumHeight(150)

    # Добавляем в groupBox (консольный вывод)
    console_layout = window.groupBox.layout()
    if console_layout is None:
        console_layout = QVBoxLayout(window.groupBox)
        window.groupBox.setLayout(console_layout)
    console_layout.addWidget(console_output)


def print_to_console(text):
    """Вывод в консоль"""
    if console_output:
        console_output.append(text)


def start_optimization():
    global current_sd, optimization_visualizer

    if not current_func:
        print_to_console("Ошибка: сначала постройте поверхность")
        return

    try:
        x0 = float(window.lineEdit_6.text())  # x0
        y0 = float(window.lineEdit_7.text())  # y0
        eps = float(window.lineEdit_8.text())  # eps
        M = int(window.lineEdit_9.text())  # M
    except ValueError:
        print_to_console("Ошибка: неверные параметры")
        return

    # Создаем визуализатор если его нет
    if optimization_visualizer is None:
        from src.core.optimization_visualizer import OptimizationVisualizer
        optimization_visualizer = OptimizationVisualizer(view)

    # Создаем оптимизатор
    current_sd = SteepestDescent(
        current_func,
        current_grad,
        [x0, y0],
        eps=eps,
        M=M
    )

    if optimization_visualizer:
        optimization_visualizer.clear()

    # Добавляем начальную точку
    z0 = current_func(x0, y0)
    start_point = to_display_coords(x0, y0, z0)
    optimization_visualizer.add_point(start_point, is_current=True)

    # Блокируем/разблокируем кнопки
    window.pushButton_2.setEnabled(False)  # Start
    window.pushButton_3.setEnabled(True)   # Step
    window.pushButton_4.setEnabled(True)   # Stop

    print_to_console(f"\n=== Запуск оптимизации ===")
    print_to_console(f"Начальная точка: [{x0}, {y0}]")
    print_to_console(f"Параметры: eps={eps}, M={M}")

    timer.start(500)


def step_optimization():
    global current_sd, optimization_visualizer

    if not current_sd:
        return

    x, done, message = current_sd.step()

    # Конвертируем в координаты отображения
    z = current_func(x[0], x[1])
    disp_point = to_display_coords(x[0], x[1], z)

    if optimization_visualizer:
        optimization_visualizer.add_point(disp_point, is_current=True)

        if len(optimization_visualizer.points) >= 2:
            prev_point = optimization_visualizer.points[-2]
            optimization_visualizer.add_arrow(prev_point, disp_point)

    # Выводим каждые 5 итераций
    if current_sd.k % 5 == 0 or done:
        print_to_console(f"Итерация {current_sd.k}: точка [{x[0]:.6f}, {x[1]:.6f}], f={z:.6f}")

    if done:
        stop_optimization()
        print_to_console(f"✅ {message}")


def stop_optimization():
    timer.stop()
    window.pushButton_2.setEnabled(True)   # Start
    window.pushButton_3.setEnabled(False)  # Step
    window.pushButton_4.setEnabled(False)  # Stop


def reset_optimization():
    stop_optimization()
    if optimization_visualizer:
        optimization_visualizer.clear()
    if current_sd:
        current_sd.reset()
    print_to_console("🔄 Сброс оптимизации")


def reset_view():
    view.setCameraPosition(distance=30, elevation=30, azimuth=30)


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


app = QApplication(sys.argv)

loader = CustomLoader()
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_path = os.path.join(current_dir, "ui", "main.ui")
window = loader.load(ui_path)

# Отладка - посмотрим какие атрибуты есть
print("Доступные кнопки:")
for attr in dir(window):
    if 'pushButton' in attr:
        print(f"  {attr}")

print("\nДоступные поля ввода:")
for attr in dir(window):
    if 'lineEdit' in attr:
        print(f"  {attr}")

# Основной 3D вид (для отрисовки поверхностей и градиента)
view = gl.GLViewWidget(parent=window.widget)
layout = window.widget.layout()
if layout is None:
    layout = QVBoxLayout(window.widget)
    window.widget.setLayout(layout)
layout.addWidget(view)
view.setCameraPosition(distance=30, elevation=30, azimuth=45)

# Сетка
grid = gl.GLGridItem()
grid.setSize(10, 10)
grid.setSpacing(1, 1)
grid.translate(0, 0, -5)  # Сетка на z=-5
view.addItem(grid)
surface_item = None

# Подключаем сигналы для отрисовки
window.pushButton.clicked.connect(update_surface)
window.comboBox.currentTextChanged.connect(on_function_changed)

# Подключаем кнопки метода (с проверкой существования)
if hasattr(window, 'pushButton_2'):
    window.pushButton_2.clicked.connect(start_optimization)  # Start
if hasattr(window, 'pushButton_3'):
    window.pushButton_3.clicked.connect(step_optimization)   # Step
if hasattr(window, 'pushButton_4'):
    window.pushButton_4.clicked.connect(stop_optimization)   # Stop
if hasattr(window, 'pushButton_5'):
    window.pushButton_5.clicked.connect(reset_optimization)  # Reset

# Настраиваем консольный вывод
setup_console()

# Таймер
timer.timeout.connect(step_optimization)

window.show()
sys.exit(app.exec())