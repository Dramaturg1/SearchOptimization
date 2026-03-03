# src/methods/gradient_descent.py
import numpy as np
from PySide6.QtWidgets import QApplication
import pyqtgraph.opengl as gl
from src.utils.point_generator import PointGenerator


class GradientDescentMethod:
    def __init__(self, view, current_func, current_zmin, current_zmax, point_item, window):
        self.view = view
        self.current_func = current_func
        self.current_zmin = current_zmin
        self.current_zmax = current_zmax
        self.point_item = point_item
        self.window = window
        self.point_generator = PointGenerator()

        self.running = False
        self.trajectory_items = []
        self.minima = []

    def set_function(self, func, zmin, zmax):
        self.current_func = func
        self.current_zmin = zmin
        self.current_zmax = zmax

    def update_bounds(self, xmin, xmax, ymin, ymax):
        """Обновление границ для генератора точек"""
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

    def gradient(self, f, x, y, h=1e-5):
        dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
        dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
        return dx, dy

    def run(self, x0=None, y0=None, eps_grad=1e-5, max_iter=100, lr=0.01, eps_pos=1e-5, eps_f=1e-6):
        # Генерация случайной точки, если не заданы координаты
        if x0 is None or y0 is None:
            x0, y0 = self.point_generator.generate_single()
            self.window.textEdit.append(f"Сгенерирована случайная начальная точка: ({x0:.4f}, {y0:.4f})")

        x, y = x0, y0
        f_prev = self.current_func(x, y)
        traj_points = []
        color = self.random_color()
        traj_item = gl.GLLinePlotItem(color=color, width=3)
        self.view.addItem(traj_item)
        self.trajectory_items.append(traj_item)

        for k in range(max_iter):
            if not self.running:
                break
            dx, dy = self.gradient(self.current_func, x, y)
            grad_norm = np.sqrt(dx ** 2 + dy ** 2)
            if grad_norm < eps_grad:
                break
            x_new = x - lr * dx
            y_new = y - lr * dy
            f_new = self.current_func(x_new, y_new)
            if f_new > f_prev:
                lr *= 0.5
                continue
            if np.sqrt((x_new - x) ** 2 + (y_new - y) ** 2) < eps_pos and abs(f_new - f_prev) < eps_f:
                x, y = x_new, y_new
                break
            x, y = x_new, y_new
            f_prev = f_new
            self.show_point(x, y)
            pos = np.array([[x, y, self.z_to_vis(self.current_func(x, y))]])
            traj_points.append(pos)
            traj_item.setData(pos=np.array(traj_points))
            QApplication.processEvents()
        return x, y, self.current_func(x, y)

    def run_multiple(self, start_points=None, eps_grad=1e-5, max_iter=100, random_count=100):
        self.running = True
        self.minima = []

        # Генерация случайных точек, если не заданы
        if start_points is None or len(start_points) == 0:
            start_points = self.point_generator.generate_multiple(random_count)
            self.window.textEdit.append(f"Сгенерировано {random_count} случайных начальных точек")

        for x0, y0 in start_points:
            if not self.running:
                break
            x, y, f = self.run(x0, y0, eps_grad, max_iter)
            self.minima.append((x, y, f))
            self.window.textEdit.append(f"Старт ({x0:.2f},{y0:.2f}): минимум найден: x={x:.5f}, y={y:.5f}, f={f:.5f}")

        if self.minima:
            global_min = min(self.minima, key=lambda t: t[2])
            self.window.textEdit.append(f"\nГлобальный минимум среди всех стартов: x={global_min[0]:.5f}, "
                                        f"y={global_min[1]:.5f}, f={global_min[2]:.5f}")

    def stop(self):
        self.running = False

    def reset(self):
        self.running = False
        for item in self.trajectory_items:
            self.view.removeItem(item)
        self.trajectory_items = []
        self.minima = []
        self.window.textEdit.append("🔄 Визуализация градиентного спуска сброшена")