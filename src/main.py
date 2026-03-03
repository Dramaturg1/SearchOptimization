from PySide6.QtWidgets import QApplication, QVBoxLayout
import pyqtgraph.opengl as gl
from src.core.CustomLoader import CustomLoader
from core.plotter import generate_surface
from src.core.surfaces import surface_data
from src.methods.gradient_descent import GradientDescentMethod
from src.methods.wolfe_method import WolfeMethod
import numpy as np
import sys
import os

surface_item = None
current_func = None
current_zmin = None
current_zmax = None
gd_method = None
wolfe_method = None


def update_surface():
    global surface_item, current_func, current_zmin, current_zmax, gd_method, wolfe_method

    try:
        xmin = float(window.lineEdit.text())
        xmax = float(window.lineEdit_2.text())
        ymin = float(window.lineEdit_3.text())
        ymax = float(window.lineEdit_4.text())
        npoints = int(window.lineEdit_5.text())
    except:
        window.textEdit.append("Ошибка ввода параметров поверхности")
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

    surface.translate(0, 0, -0.1)
    view.addItem(surface)
    surface_item = surface

    current_func = func
    current_zmin = zmin
    current_zmax = zmax

    if gd_method:
        gd_method.set_function(current_func, current_zmin, current_zmax)
        gd_method.update_bounds(xmin, xmax, ymin, ymax)  # Добавлено
    if wolfe_method:
        wolfe_method.set_function(current_func, current_zmin, current_zmax)
        wolfe_method.update_bounds(xmin, xmax, ymin, ymax)  # Добавлено

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
        window.textEdit.append("Ошибка параметров градиентного спуска")
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


def wolfe_optimization():
    global wolfe_method

    if current_func is None:
        window.textEdit.append("Сначала построй поверхность")
        return

    try:
        # Пытаемся прочитать координаты из полей ввода
        x_start = float(window.lineEdit_9.text())  # x0 для Вульфа
        y_start = float(window.lineEdit_10.text())  # y0 для Вульфа
        eps = float(window.lineEdit_12.text())  # eps для Вульфа
        max_iter = int(window.lineEdit_11.text())  # max_iter для Вульфа
        start_points = [(x_start, y_start)]

        # Если все поля заполнены корректно, используем введенные значения
        window.textEdit.append("=" * 60)
        window.textEdit.append("ЗАПУСК МЕТОДА ВУЛЬФА")
        window.textEdit.append(f"Параметры: x0={x_start}, y0={y_start}, eps={eps}, max_iter={max_iter}")
        window.textEdit.append("=" * 60)

    except ValueError:
        # Если ошибка ввода, используем случайные точки
        window.textEdit.append("Ошибка ввода параметров метода Вульфа. Используются случайные точки")

        # Получаем границы текущей поверхности
        xmin = float(window.lineEdit.text())
        xmax = float(window.lineEdit_2.text())
        ymin = float(window.lineEdit_3.text())
        ymax = float(window.lineEdit_4.text())

        # Количество случайных точек для запуска
        N = 5

        # Генерируем случайные точки
        start_points = [
            (np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)) for _ in range(N)
        ]

        # Стандартные параметры для метода
        eps = 1e-6
        max_iter = 100

        window.textEdit.append("=" * 60)
        window.textEdit.append("ЗАПУСК МЕТОДА ВУЛЬФА СО СЛУЧАЙНЫМИ ТОЧКАМИ")
        window.textEdit.append(f"Параметры: eps={eps}, max_iter={max_iter}, точек={N}")
        window.textEdit.append("Случайные точки:")
        for i, (x, y) in enumerate(start_points):
            window.textEdit.append(f"  Точка {i + 1}: ({x:.4f}, {y:.4f})")
        window.textEdit.append("=" * 60)

    # Создаем или обновляем объект метода Вульфа
    if wolfe_method is None:
        wolfe_method = WolfeMethod(view, current_func, current_zmin, current_zmax, point_item, window)
    else:
        wolfe_method.reset()
        wolfe_method.set_function(current_func, current_zmin, current_zmax)

    # Запускаем метод Вульфа с несколькими начальными точками
    wolfe_method.run_multiple(start_points, eps, max_iter, random_count=100)


def wolfe_step():
    """Пошаговый режим метода Вульфа"""
    global wolfe_method

    if current_func is None:
        window.textEdit.append("Сначала построй поверхность")
        return

    if wolfe_method is None:
        try:
            # Пытаемся прочитать координаты из полей ввода
            x_start = float(window.lineEdit_9.text())
            y_start = float(window.lineEdit_10.text())
            eps = float(window.lineEdit_12.text())
            max_iter = int(window.lineEdit_11.text())
        except ValueError:
            # Если ошибка ввода, используем случайную точку
            window.textEdit.append("Ошибка ввода параметров. Используется случайная начальная точка")

            # Получаем границы текущей поверхности
            xmin = float(window.lineEdit.text())
            xmax = float(window.lineEdit_2.text())
            ymin = float(window.lineEdit_3.text())
            ymax = float(window.lineEdit_4.text())

            # Генерируем случайную точку
            x_start = np.random.uniform(xmin, xmax)
            y_start = np.random.uniform(ymin, ymax)
            eps = 1e-6
            max_iter = 50

        # Создаем объект метода Вульфа
        wolfe_method = WolfeMethod(view, current_func, current_zmin, current_zmax, point_item, window)
        wolfe_method.set_function(current_func, current_zmin, current_zmax)

        # Запускаем пошаговый режим
        wolfe_method.run_step_mode(x_start, y_start, eps, max_iter)
    else:
        # Делаем следующий шаг
        wolfe_method.step()
def stop_gd():
    global gd_method
    if gd_method:
        gd_method.stop()

def reset_gd():
    global gd_method
    if gd_method:
        gd_method.reset()

def stop_wolfe():
    global wolfe_method
    if wolfe_method:
        wolfe_method.stop()
        window.textEdit.append("Метод Вульфа остановлен")


def reset_wolfe():
    global wolfe_method
    if wolfe_method:
        wolfe_method.reset()
    window.textEdit.append("Визуализация метода Вульфа очищена")


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
axis.setSize(5, 5, 5)
view.addItem(axis)

point_item = gl.GLScatterPlotItem(
    size=15,
    color=(1, 0, 0, 1)
)
point_item.setGLOptions('opaque')
view.addItem(point_item)

# Устанавливаем значения по умолчанию
window.lineEdit_6.setText("2")  # x0 для градиентного спуска
window.lineEdit_7.setText("2")  # y0 для градиентного спуска
window.lineEdit_8.setText("1e-5")  # eps для градиентного спуска
window.lineEdit_9.setText("2")  # x0 для метода Вульфа
window.lineEdit_10.setText("2")  # y0 для метода Вульфа
window.lineEdit_11.setText("50")  # max_iter для метода Вульфа
window.lineEdit_12.setText("1e-6")  # eps для метода Вульфа

# Подключаем сигналы для построения поверхности
window.pushButton.clicked.connect(update_surface)
window.comboBox.currentTextChanged.connect(on_function_changed)

# Подключаем сигналы для градиентного спуска (первая вкладка)
window.pushButton_2.clicked.connect(gradient_descent)  # Start
window.pushButton_4.clicked.connect(stop_gd)  # Stop (общая остановка)
window.pushButton_5.clicked.connect(reset_gd)  # Reset (общий сброс)

# Подключаем сигналы для метода Вульфа (вторая вкладка)
window.pushButton_6.clicked.connect(wolfe_optimization)  # Start
window.pushButton_7.clicked.connect(stop_wolfe)  # Stop
window.pushButton_8.clicked.connect(reset_wolfe)  # Reset
window.pushButton_9.clicked.connect(wolfe_step)  # Step

window.show()
sys.exit(app.exec())