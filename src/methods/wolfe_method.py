# src/methods/wolfe_method.py
import numpy as np
from PySide6.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import time
from src.utils.point_generator import PointGenerator


class WolfeMethod:
    def __init__(self, view, current_func, current_zmin, current_zmax, point_item, window):
        self.view = view
        self.current_func = current_func
        self.current_zmin = current_zmin
        self.current_zmax = current_zmax
        self.point_item = point_item
        self.window = window
        self.point_generator = PointGenerator()

        self.running = False
        self.step_mode = False
        self.current_iteration = 0
        self.max_iterations = 0
        self.x0 = 0
        self.y0 = 0
        self.x_opt = None
        self.y_opt = None
        self.direction = None

        self.trajectory_items = []
        self.start_points_items = []
        self.end_points_items = []
        self.minima = []
        self.current_point_item = None
        self.trajectory_points = []
        self.trajectory_line = None

    def set_function(self, func, zmin, zmax):
        self.current_func = func
        self.current_zmin = zmin
        self.current_zmax = zmax

    def update_bounds(self, xmin, xmax, ymin, ymax):
        self.point_generator.update_bounds(xmin, xmax, ymin, ymax)

    def z_to_vis(self, z):
        if self.current_zmax == self.current_zmin:
            return self.current_zmax
        return (z - self.current_zmin) / (self.current_zmax - self.current_zmin) * 10

    def show_point(self, x, y):
        if self.current_func is None:
            return
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])
        self.point_item.setData(pos=pos)

    def random_color(self):
        return (np.random.rand(), np.random.rand(), np.random.rand(), 1.0)

    def show_start_point(self, x, y, color=None):
        if self.current_func is None:
            return
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])

        if color is None:
            color = (0, 1, 0, 1)

        point = gl.GLScatterPlotItem(
            pos=pos,
            size=15,
            color=color
        )
        self.view.addItem(point)
        self.start_points_items.append(point)

    def show_end_point(self, x, y, color=None):
        if self.current_func is None:
            return
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])

        if color is None:
            color = (1, 0, 0, 1)

        point = gl.GLScatterPlotItem(
            pos=pos,
            size=15,
            color=color
        )
        self.view.addItem(point)
        self.end_points_items.append(point)

    def show_current_point(self, x, y, color=(0, 1, 0, 1)):
        if self.current_func is None:
            return
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])

        if self.current_point_item:
            self.view.removeItem(self.current_point_item)

        self.current_point_item = gl.GLScatterPlotItem(
            pos=pos,
            size=20,
            color=color
        )
        self.view.addItem(self.current_point_item)

    def add_trajectory(self, points, color=None):
        if len(points) < 2:
            return

        if color is None:
            color = self.random_color()

        trajectory_line = gl.GLLinePlotItem(
            pos=np.array(points),
            color=color,
            width=2
        )
        self.view.addItem(trajectory_line)
        self.trajectory_items.append(trajectory_line)

        return trajectory_line

    def add_trajectory_point(self, x, y):
        if self.current_func is None:
            return
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        self.trajectory_points.append([x, y, z_vis])

        if len(self.trajectory_points) > 1:
            if self.trajectory_line:
                self.view.removeItem(self.trajectory_line)

            self.trajectory_line = gl.GLLinePlotItem(
                pos=np.array(self.trajectory_points),
                color=(1, 0.5, 0, 1),
                width=3
            )
            self.view.addItem(self.trajectory_line)

    def extract_quadratic_coefficients(self, func, x0, y0, h=0.1):
        d2x = (func(x0 + h, y0) - 2 * func(x0, y0) + func(x0 - h, y0)) / (h ** 2)
        d2y = (func(x0, y0 + h) - 2 * func(x0, y0) + func(x0, y0 - h)) / (h ** 2)
        d2xy = (func(x0 + h, y0 + h) - func(x0 + h, y0 - h) - func(x0 - h, y0 + h) + func(x0 - h, y0 - h)) / (
                4 * h ** 2)

        dx = (func(x0 + h, y0) - func(x0 - h, y0)) / (2 * h)
        dy = (func(x0, y0 + h) - func(x0, y0 - h)) / (2 * h)

        Q = np.array([[d2x, d2xy], [d2xy, d2y]])
        c = np.array([dx, dy]) - Q @ np.array([x0, y0])
        const = func(x0, y0) - 0.5 * (x0 ** 2 * d2x + 2 * x0 * y0 * d2xy + y0 ** 2 * d2y) - c[0] * x0 - c[1] * y0

        return {'Q': Q, 'c': c, 'const': const}

    def wolfe_method(self, Q, c, const=0):
        try:
            x_opt = np.linalg.solve(Q, -c)
            return x_opt
        except np.linalg.LinAlgError:
            Q_pinv = np.linalg.pinv(Q)
            x_opt = Q_pinv @ (-c)
            return x_opt

    def run_step_mode(self, x0=None, y0=None, eps=1e-6, max_iter=100):
        if x0 is None or y0 is None:
            x0, y0 = self.point_generator.generate_single()
            self.window.textEdit.append(f"Сгенерирована случайная начальная точка: ({x0:.4f}, {y0:.4f})")

        self.step_mode = True
        self.running = True
        self.current_iteration = 0
        self.max_iterations = max_iter
        self.x0 = x0
        self.y0 = y0

        coeffs = self.extract_quadratic_coefficients(self.current_func, x0, y0)
        Q = coeffs['Q']
        c = coeffs['c']

        x_opt = self.wolfe_method(Q, c)
        self.x_opt = x_opt[0]
        self.y_opt = x_opt[1]

        self.direction = np.array([self.x_opt - x0, self.y_opt - y0])

        self.trajectory_points = []
        self.show_current_point(x0, y0, (0, 1, 0, 1))
        self.add_trajectory_point(x0, y0)

        self.window.textEdit.append(f"Пошаговый режим метода Вульфа")
        self.window.textEdit.append(f"Начальная точка: ({x0:.4f}, {y0:.4f})")
        self.window.textEdit.append(f"Целевая точка: ({self.x_opt:.4f}, {self.y_opt:.4f})")
        self.window.textEdit.append(f"Всего шагов: {max_iter}. Нажмите Step для следующего шага")

    def step(self):
        if not self.step_mode or not self.running:
            return

        if self.current_iteration >= self.max_iterations:
            self.window.textEdit.append("Достигнуто максимальное число итераций")
            return

        self.current_iteration += 1

        t = self.current_iteration / self.max_iterations

        x_current = self.x0 + self.direction[0] * t
        y_current = self.y0 + self.direction[1] * t

        self.show_point(x_current, y_current)
        self.add_trajectory_point(x_current, y_current)

        color = (t, 1 - t, 0, 1)
        self.show_current_point(x_current, y_current, color)

        f_current = self.current_func(x_current, y_current)
        self.window.textEdit.append(f"Шаг {self.current_iteration}/{self.max_iterations}: "
                                    f"x={x_current:.6f}, y={y_current:.6f}, f={f_current:.6f}")

        if self.current_iteration == self.max_iterations:
            if len(self.trajectory_points) > 1:
                color = self.random_color()

                permanent_line = gl.GLLinePlotItem(
                    pos=np.array(self.trajectory_points),
                    color=color,
                    width=2
                )
                self.view.addItem(permanent_line)
                self.trajectory_items.append(permanent_line)
                self.show_start_point(self.x0, self.y0, (0, 1, 0, 1))
                self.show_end_point(self.x_opt, self.y_opt, (1, 0, 0, 1))

                if self.trajectory_line:
                    self.view.removeItem(self.trajectory_line)
                    self.trajectory_line = None

            self.show_point(self.x_opt, self.y_opt)
            self.show_current_point(self.x_opt, self.y_opt, (1, 0, 0, 1))
            self.window.textEdit.append(f"Достигнут оптимум!")

    def run(self, x0=None, y0=None, eps=1e-6, max_iter=100):
        if x0 is None or y0 is None:
            x0, y0 = self.point_generator.generate_single()
            self.window.textEdit.append(f"Сгенерирована случайная начальная точка: ({x0:.4f}, {y0:.4f})")
        coeffs = self.extract_quadratic_coefficients(self.current_func, x0, y0)
        Q = coeffs['Q']
        c = coeffs['c']
        x_opt = self.wolfe_method(Q, c)
        color = self.random_color()
        trajectory_line = gl.GLLinePlotItem(color=color, width=2)
        self.view.addItem(trajectory_line)
        self.trajectory_items.append(trajectory_line)
        trajectory_points = []
        z0 = self.current_func(x0, y0)
        z0_vis = self.z_to_vis(z0)
        trajectory_points.append([x0, y0, z0_vis])
        self.show_start_point(x0, y0, (0, 1, 0, 1))

        direction = x_opt - np.array([x0, y0])

        for iteration in range(max_iter):
            if not self.running:
                break

            t = (iteration + 1) / max_iter
            x_current = x0 + direction[0] * t
            y_current = y0 + direction[1] * t

            z_current = self.current_func(x_current, y_current)
            z_vis = self.z_to_vis(z_current)
            trajectory_points.append([x_current, y_current, z_vis])
            trajectory_line.setData(pos=np.array(trajectory_points))
            self.show_point(x_current, y_current)
            QApplication.processEvents()
            time.sleep(0.02)
        z_opt = self.current_func(x_opt[0], x_opt[1])
        z_opt_vis = self.z_to_vis(z_opt)
        trajectory_points.append([x_opt[0], x_opt[1], z_opt_vis])
        trajectory_line.setData(pos=np.array(trajectory_points))
        self.show_end_point(x_opt[0], x_opt[1], (1, 0, 0, 1))
        self.show_point(x_opt[0], x_opt[1])

        return x_opt[0], x_opt[1], self.current_func(x_opt[0], x_opt[1])

    def run_multiple(self, start_points=None, eps=1e-6, max_iter=100, random_count=100):
        if start_points is None or len(start_points) == 0:
            start_points = self.point_generator.generate_multiple(random_count)
            self.window.textEdit.append(f"Сгенерировано {random_count} случайных начальных точек")
            for i, (x, y) in enumerate(start_points):
                self.window.textEdit.append(f"  Точка {i + 1}: ({x:.4f}, {y:.4f})")

        self.running = True
        self.minima = []

        for i, (x0, y0) in enumerate(start_points):
            if not self.running:
                break

            self.window.textEdit.append(f"\nЗапуск {i + 1} из {len(start_points)}")
            x, y, f = self.run(x0, y0, eps, max_iter)
            self.minima.append((x, y, f))

            if i < len(start_points) - 1:
                time.sleep(0.5)

        if self.minima:
            global_min = min(self.minima, key=lambda t: t[2])
            self.window.textEdit.append(f"\nГлобальный минимум: x={global_min[0]:.8f}, "
                                        f"y={global_min[1]:.8f}, f={global_min[2]:.8f}")

            self.show_point(global_min[0], global_min[1])

    def stop(self):
        self.running = False
        self.step_mode = False

    def reset(self):
        self.running = False
        self.step_mode = False
        self.current_iteration = 0

        for item in self.trajectory_items:
            self.view.removeItem(item)
        self.trajectory_items = []
        for item in self.start_points_items:
            self.view.removeItem(item)
        self.start_points_items = []
        for item in self.end_points_items:
            self.view.removeItem(item)
        self.end_points_items = []
        if self.trajectory_line:
            self.view.removeItem(self.trajectory_line)
            self.trajectory_line = None
        if self.current_point_item:
            self.view.removeItem(self.current_point_item)
            self.current_point_item = None

        self.trajectory_points = []
        self.minima = []
        self.window.textEdit.append("Визуализация метода Вульфа сброшена")