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

# Ограничения
DEFAULT_CONSTRAINTS = {
    "A": np.array([[1, 1], [1, 2], [-1, 0], [0, -1]]),
    "b": np.array([2, 3, 0, 0])
}

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
        gd_method.update_bounds(xmin, xmax, ymin, ymax)

    if wolfe_method:
        wolfe_method.set_function(current_func, current_zmin, current_zmax)
        wolfe_method.update_bounds(xmin, xmax, ymin, ymax)
        if name == "Тестовая функция (методичка)":
            wolfe_method.set_constraints(DEFAULT_CONSTRAINTS["A"], DEFAULT_CONSTRAINTS["b"])
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

    reset_wolfe()
    reset_gd()


def gradient_descent():
    global gd_method

    if current_func is None:
        window.textEdit.append("Сначала постройте поверхность")
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
        window.textEdit.append("Сначала постройте поверхность")
        return

    try:

        x_start = float(window.lineEdit_9.text())   # x0 для Вульфа
        y_start = float(window.lineEdit_10.text())  # y0 для Вульфа
        eps = float(window.lineEdit_12.text())      # точность
        max_iter = int(window.lineEdit_11.text())   # макс итераций

        start_points = [(x_start, y_start)]

        window.textEdit.append("Метод Вулфа:")
        window.textEdit.append(f"Параметры:")
        window.textEdit.append(f"  Начальная точка: ({x_start:.4f}, {y_start:.4f})")
        window.textEdit.append(f"  Точность: {eps}")
        window.textEdit.append(f"  Макс. итераций: {max_iter}")


        if window.comboBox.currentText() == "Тестовая функция (методичка)":
            window.textEdit.append("\nОграничения задачи:")
            window.textEdit.append("  x1 + x2 ≤ 2")
            window.textEdit.append("  x1 + 2x2 ≤ 3")
            window.textEdit.append("  x1 ≥ 0, x2 ≥ 0")

    except ValueError as e:
        window.textEdit.append(f"\nОшибка ввода параметров: {e}")
        window.textEdit.append("Используются случайные начальные точки")

        xmin = float(window.lineEdit.text())
        xmax = float(window.lineEdit_2.text())
        ymin = float(window.lineEdit_3.text())
        ymax = float(window.lineEdit_4.text())
        N = 50
        start_points = [
            (np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)) for _ in range(N)
        ]
        eps = 1e-6
        max_iter = 100

        window.textEdit.append("Метод Вулфа со случайными точками: ")
        window.textEdit.append(f"Параметры:")
        window.textEdit.append(f"  Точность: {eps}")
        window.textEdit.append(f"  Макс. итераций: {max_iter}")
        window.textEdit.append(f"  Количество точек: {N}")

    if wolfe_method is None:
        wolfe_method = WolfeMethod(view, current_func, current_zmin, current_zmax, point_item, window)
    else:
        wolfe_method.reset()
        wolfe_method.set_function(current_func, current_zmin, current_zmax)

    if window.comboBox.currentText() == "Тестовая функция (методичка)":
        wolfe_method.set_constraints(DEFAULT_CONSTRAINTS["A"], DEFAULT_CONSTRAINTS["b"])

    wolfe_method.run_multiple(start_points, eps, max_iter, len(start_points))


def wolfe_step():
    global wolfe_method

    if current_func is None:
        window.textEdit.append("Сначала постройте поверхность")
        return

    if wolfe_method is None:
        try:
            x_start = float(window.lineEdit_9.text())
            y_start = float(window.lineEdit_10.text())
            eps = float(window.lineEdit_12.text())
            max_iter = int(window.lineEdit_11.text())

            wolfe_method = WolfeMethod(view, current_func, current_zmin, current_zmax, point_item, window)
            wolfe_method.set_function(current_func, current_zmin, current_zmax)

            if window.comboBox.currentText() == "Тестовая функция (методичка)":
                wolfe_method.set_constraints(DEFAULT_CONSTRAINTS["A"], DEFAULT_CONSTRAINTS["b"])

            wolfe_method.run_step_mode(x_start, y_start, eps, max_iter)

        except ValueError:
            window.textEdit.append("Ошибка ввода параметров. Используется случайная начальная точка")
            xmin = float(window.lineEdit.text())
            xmax = float(window.lineEdit_2.text())
            ymin = float(window.lineEdit_3.text())
            ymax = float(window.lineEdit_4.text())
            x_start = np.random.uniform(xmin, xmax)
            y_start = np.random.uniform(ymin, ymax)
            eps = 1e-6
            max_iter = 50

            wolfe_method = WolfeMethod(view, current_func, current_zmin, current_zmax, point_item, window)
            wolfe_method.set_function(current_func, current_zmin, current_zmax)

            # Установка ограничений для тестовой функции
            if window.comboBox.currentText() == "Тестовая функция (методичка)":
                wolfe_method.set_constraints(DEFAULT_CONSTRAINTS["A"], DEFAULT_CONSTRAINTS["b"])

            wolfe_method.run_step_mode(x_start, y_start, eps, max_iter)
    else:
        if hasattr(wolfe_method, 'step_mode') and wolfe_method.step_mode:
            wolfe_method.step()
        else:
            window.textEdit.append("Пошаговый режим не активен")


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


def reset_wolfe():
    global wolfe_method
    if wolfe_method:
        wolfe_method.reset()



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

window.lineEdit_6.setText("2")      # x0 для градиентного спуска
window.lineEdit_7.setText("2")      # y0 для градиентного спуска
window.lineEdit_8.setText("1e-5")   # точность градиентного спуска
window.lineEdit_9.setText("2")      # x0 для метода Вулфа
window.lineEdit_10.setText("2")     # y0 для метода Вулфа
window.lineEdit_11.setText("50")    # макс итераций для Вулфа
window.lineEdit_12.setText("1e-6")  # точность для Вулфа

# Подключение сигналов
window.pushButton.clicked.connect(update_surface)  # Построить
window.comboBox.currentTextChanged.connect(on_function_changed)  # Смена функции

# Градиентный спуск
window.pushButton_2.clicked.connect(gradient_descent)  # Старт
window.pushButton_4.clicked.connect(stop_gd)  # Стоп
window.pushButton_5.clicked.connect(reset_gd)  # Сброс

# Метод Вулфа
window.pushButton_6.clicked.connect(wolfe_optimization)  # Оптимизация
window.pushButton_7.clicked.connect(stop_wolfe)  # Стоп
window.pushButton_8.clicked.connect(reset_wolfe)  # Сброс
window.pushButton_9.clicked.connect(wolfe_step)  # Шаг

window.comboBox.setCurrentText("Тестовая функция (методичка)")
on_function_changed()

window.show()
sys.exit(app.exec())