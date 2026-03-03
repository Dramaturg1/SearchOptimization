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

        self.constraints = []
        self.A = None
        self.b = None
        self.has_constraints = False

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

    def set_constraints(self, A, b):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.constraints = [(self.A[i], self.b[i]) for i in range(len(self.b))]
        self.has_constraints = True

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

        point = gl.GLScatterPlotItem(pos=pos, size=15, color=color)
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

        point = gl.GLScatterPlotItem(pos=pos, size=15, color=color)
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

        self.current_point_item = gl.GLScatterPlotItem(pos=pos, size=20, color=color)
        self.view.addItem(self.current_point_item)

    def add_trajectory(self, points, color=None):
        if len(points) < 2:
            return

        if color is None:
            color = self.random_color()

        trajectory_line = gl.GLLinePlotItem(pos=np.array(points), color=color, width=2)
        self.view.addItem(trajectory_line)
        self.trajectory_items.append(trajectory_line)
        return trajectory_line

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

    def print_tableau(self, tableau, basis, iteration):
        self.window.textEdit.append(f"\nИтерация {iteration}")
        self.window.textEdit.append("Симплекс-таблица:")

        header = "     "
        for j in range(tableau.shape[1] - 1):
            header += f"x{j:2d} "
        header += "  RHS"
        self.window.textEdit.append(header)

        for i in range(tableau.shape[0]):
            if i == 0:
                row = "  z  "
            else:
                row = f" b{i}  "
            for j in range(tableau.shape[1]):
                row += f"{tableau[i, j]:6.2f} "
            self.window.textEdit.append(row)

        self.window.textEdit.append(f"Базис: {basis}")

    def print_solution(self, x, u, v, s, iteration):
        self.window.textEdit.append(f"\nРешение после итерации {iteration}")
        self.window.textEdit.append(f"x = [{x[0]:.6f}, {x[1]:.6f}]")
        self.window.textEdit.append(f"u = [{u[0]:.6f}, {u[1]:.6f}, {u[2]:.6f}, {u[3]:.6f}]")
        self.window.textEdit.append(f"v = [{v[0]:.6f}, {v[1]:.6f}]")
        self.window.textEdit.append(f"s = [{s[0]:.6f}, {s[1]:.6f}, {s[2]:.6f}, {s[3]:.6f}]")

    def create_simplex_tableau(self, Q, c, A, b):
        n = len(c)
        m = len(b)

        rows = 1 + n + m
        cols = 2 * n + 2 * m + n + 1

        tableau = np.zeros((rows, cols))
        for j in range(n):
            tableau[0, 2 * n + 2 * m + j] = 1

        for i in range(n):
            for j in range(n):
                tableau[i + 1, j] = Q[i, j]
            for j in range(m):
                tableau[i + 1, n + j] = A[j, i]
            tableau[i + 1, n + m + i] = -1
            tableau[i + 1, 2 * n + 2 * m + i] = 1
            tableau[i + 1, -1] = -c[i]

        for i in range(m):
            for j in range(n):
                tableau[n + i + 1, j] = A[i, j]
            tableau[n + i + 1, n + m + n + i] = 1
            tableau[n + i + 1, -1] = b[i]

        basis = list(range(2 * n + 2 * m, 2 * n + 2 * m + n))

        self.window.textEdit.append("\nНачальная симплекс-таблица")
        self.print_tableau(tableau, basis, 0)

        return tableau, basis

    def modified_simplex(self, tableau, basis, max_iter=100):
        rows, cols = tableau.shape
        n_vars = 2
        m_constr = 4

        for iteration in range(max_iter):
            self.print_tableau(tableau, basis, iteration + 1)

            if np.all(tableau[0, :-1] <= 1e-10):
                self.window.textEdit.append(f"Оптимальность достигнута на итерации {iteration}")
                break

            entering = np.argmax(tableau[0, :-1])
            self.window.textEdit.append(f"Ведущий столбец: {entering}")

            if np.all(tableau[1:, entering] <= 1e-10):
                self.window.textEdit.append("Задача неограничена")
                break

            ratios = []
            valid_rows = []
            for i in range(1, rows):
                if tableau[i, entering] > 1e-10:
                    ratio = tableau[i, -1] / tableau[i, entering]
                    ratios.append(ratio)
                    valid_rows.append(i)

            if not valid_rows:
                break

            leaving = valid_rows[np.argmin(ratios)]
            self.window.textEdit.append(f"Ведущая строка: {leaving}")

            pivot = tableau[leaving, entering]
            tableau[leaving, :] /= pivot

            for i in range(rows):
                if i != leaving:
                    factor = tableau[i, entering]
                    tableau[i, :] -= factor * tableau[leaving, :]

            basis[leaving - 1] = entering

            x = np.zeros(n_vars)
            u = np.zeros(m_constr)
            v = np.zeros(n_vars)
            s = np.zeros(m_constr)

            for i, var in enumerate(basis):
                if var < n_vars:
                    x[var] = tableau[i + 1, -1]
                elif var < n_vars + m_constr:
                    u[var - n_vars] = tableau[i + 1, -1]
                elif var < 2 * n_vars + m_constr:
                    v[var - (n_vars + m_constr)] = tableau[i + 1, -1]
                elif var < 2 * n_vars + 2 * m_constr:
                    s[var - (2 * n_vars + m_constr)] = tableau[i + 1, -1]

            self.print_solution(x, u, v, s, iteration + 1)

        x = np.zeros(n_vars)
        u = np.zeros(m_constr)
        v = np.zeros(n_vars)
        s = np.zeros(m_constr)

        for i, var in enumerate(basis):
            if var < n_vars:
                x[var] = tableau[i + 1, -1]
            elif var < n_vars + m_constr:
                u[var - n_vars] = tableau[i + 1, -1]
            elif var < 2 * n_vars + m_constr:
                v[var - (n_vars + m_constr)] = tableau[i + 1, -1]
            elif var < 2 * n_vars + 2 * m_constr:
                s[var - (2 * n_vars + m_constr)] = tableau[i + 1, -1]

        return x, u, v, s, iteration

    def wolfe_method(self, Q, c, A=None, b=None):
        n = 2

        if A is None or b is None or not self.has_constraints:
            try:
                x_opt = np.linalg.solve(Q, -c)
                return x_opt
            except:
                Q_pinv = np.linalg.pinv(Q)
                x_opt = Q_pinv @ (-c)
                return x_opt

        tableau, basis = self.create_simplex_tableau(Q, c, A, b)
        x_full, u, v, s, iterations = self.modified_simplex(tableau, basis)

        x_opt = x_full[:2]
        f_opt = 0.5 * x_opt @ Q @ x_opt + c @ x_opt
        self.window.textEdit.append(f"f* = {f_opt:.6f}")

        return x_opt

    def run(self, x0=None, y0=None, eps=1e-6, max_iter=100):
        if x0 is None or y0 is None:
            x0, y0 = self.point_generator.generate_single()

        coeffs = self.extract_quadratic_coefficients(self.current_func, x0, y0)
        Q = coeffs['Q']
        c = coeffs['c']


        x_opt = self.wolfe_method(Q, c, self.A, self.b)

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

        for iteration in range(min(50, max_iter)):
            if not self.running:
                break

            t = (iteration + 1) / 50
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

    def run_multiple(self, start_points=None, eps=1e-6, max_iter=100, random_count=50):
        if start_points is None or len(start_points) == 0:
            start_points = []
            for _ in range(random_count):
                x, y = self.point_generator.generate_single()
                start_points.append((x, y))

        self.running = True
        self.minima = []

        for i, (x0, y0) in enumerate(start_points):
            if not self.running:
                break

            x, y, f = self.run(x0, y0, eps, max_iter)
            self.minima.append((x, y, f))

            if i < len(start_points) - 1:
                time.sleep(0.5)

        if self.minima:
            global_min = min(self.minima, key=lambda t: t[2])
            self.window.textEdit.append(f"\nГЛОБАЛЬНЫЙ МИНИМУМ:")
            self.window.textEdit.append(f"x = {global_min[0]:.8f}")
            self.window.textEdit.append(f"y = {global_min[1]:.8f}")
            self.window.textEdit.append(f"f = {global_min[2]:.8f}")

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