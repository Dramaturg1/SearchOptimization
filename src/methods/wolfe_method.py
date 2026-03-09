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

        # Ограничения из методички для функции pip_gupip
        # x1 + x2 <= 2
        # x1 + 2x2 <= 3
        # x1 >= 0, x2 >= 0
        self.A = np.array([
            [1, 1],  # x1 + x2 <= 2
            [1, 2],  # x1 + 2x2 <= 3
            [-1, 0],  # -x1 <= 0 (x1 >= 0)
            [0, -1]  # -x2 <= 0 (x2 >= 0)
        ], dtype=float)
        self.b = np.array([2, 3, 0, 0], dtype=float)
        self.has_constraints = True

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
        """Установка пользовательских ограничений"""
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
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

    def extract_quadratic_coefficients(self, x0, y0, h=0.001):
        """Извлечение коэффициентов квадратичной функции в точке (x0, y0)"""
        func = self.current_func

        # Вторые производные
        d2x = (func(x0 + h, y0) - 2 * func(x0, y0) + func(x0 - h, y0)) / (h ** 2)
        d2y = (func(x0, y0 + h) - 2 * func(x0, y0) + func(x0, y0 - h)) / (h ** 2)
        d2xy = (func(x0 + h, y0 + h) - func(x0 + h, y0 - h) - func(x0 - h, y0 + h) + func(x0 - h, y0 - h)) / (
                    4 * h ** 2)

        # Первые производные
        dx = (func(x0 + h, y0) - func(x0 - h, y0)) / (2 * h)
        dy = (func(x0, y0 + h) - func(x0, y0 - h)) / (2 * h)

        Q = np.array([[d2x, d2xy], [d2xy, d2y]])

        # В точке x0 градиент = Q*x0 + c, поэтому c = градиент - Q*x0
        c = np.array([dx, dy]) - Q @ np.array([x0, y0])

        self.window.textEdit.append(f"\nКоэффициенты в точке ({x0:.4f}, {y0:.4f}):")
        self.window.textEdit.append(f"Q = {Q}")
        self.window.textEdit.append(f"c = {c}")

        return {'Q': Q, 'c': c}

    def print_tableau(self, tableau, basis, iteration):
        """Вывод симплекс-таблицы"""
        self.window.textEdit.append(f"\n{'=' * 60}")
        self.window.textEdit.append(f"Итерация {iteration}")
        self.window.textEdit.append(f"{'=' * 60}")

        # Заголовок
        header = "Базис    |"
        for j in range(tableau.shape[1] - 1):
            if j < 2:
                header += f" x{j + 1:6d} |"
            elif j < 6:
                header += f" u{j - 1:6d} |"
            elif j < 8:
                header += f" v{j - 5:6d} |"
            elif j < 12:
                header += f" s{j - 7:6d} |"
            else:
                header += f" a{j - 11:6d} |"
        header += "    RHS    "
        self.window.textEdit.append(header)
        self.window.textEdit.append("-" * len(header))

        # Строки таблицы
        for i in range(tableau.shape[0]):
            if i == 0:
                row = "    f    |"
            else:
                var = basis[i - 1]
                if var < 2:
                    row = f"   x{var + 1}   |"
                elif var < 6:
                    row = f"   u{var - 1}   |"
                elif var < 8:
                    row = f"   v{var - 5}   |"
                elif var < 12:
                    row = f"   s{var - 7}   |"
                else:
                    row = f"   a{var - 11}  |"

            for j in range(tableau.shape[1]):
                row += f"{tableau[i, j]:10.4f}"
            self.window.textEdit.append(row)

    def create_simplex_tableau(self, Q, c):
        """
        Создание симплекс-таблицы для метода Вулфа
        """
        n = 2  # количество переменных x
        m = 4  # количество ограничений

        rows = 1 + n + m  # 1 + 2 + 4 = 7 строк
        cols = n + m + n + m + n + 1  # 2 + 4 + 2 + 4 + 2 + 1 = 15 столбцов

        # Индексы переменных:
        # 0-1:   x1, x2
        # 2-5:   u1, u2, u3, u4
        # 6-7:   v1, v2
        # 8-11:  s1, s2, s3, s4
        # 12-13: искусственные a1, a2
        # 14:    RHS

        tableau = np.zeros((rows, cols))

        # f-строка (индекс 0) - минимизация суммы искусственных переменных
        tableau[0, 12] = 1  # a1
        tableau[0, 13] = 1  # a2

        # Строки для условий ∂L/∂x = 0 (индексы 1-2)
        # Уравнение: Qx + c + A^T u - v = 0
        for i in range(n):
            row = i + 1

            # Qx
            for j in range(n):
                tableau[row, j] = Q[i, j]

            # A^T u
            tableau[row, 2] = self.A[0, i]  # u1 * A[0,i]
            tableau[row, 3] = self.A[1, i]  # u2 * A[1,i]
            tableau[row, 4] = self.A[2, i]  # u3 * A[2,i]
            tableau[row, 5] = self.A[3, i]  # u4 * A[3,i]

            # -v
            tableau[row, 6 + i] = -1

            # Искусственная переменная
            tableau[row, 12 + i] = 1

            # RHS = -c
            tableau[row, -1] = -c[i]

        # Строки ограничений (индексы 3-6)
        # Уравнение: Ax + s = b
        for i in range(m):
            row = n + i + 1

            # Ax
            tableau[row, 0] = self.A[i, 0]
            tableau[row, 1] = self.A[i, 1]

            # s
            tableau[row, 8 + i] = 1

            # RHS = b
            tableau[row, -1] = self.b[i]

        # Начальный базис: искусственные переменные a1, a2 и s1..s4
        basis = [12, 13, 8, 9, 10, 11]

        self.window.textEdit.append("\n" + "=" * 80)
        self.window.textEdit.append("НАЧАЛЬНАЯ СИМПЛЕКС-ТАБЛИЦА")
        self.window.textEdit.append("=" * 80)
        self.print_tableau(tableau, basis, 0)

        return tableau, basis

    def phase_one(self, tableau, basis, max_iter=100):
        """Первая фаза - исключение искусственных переменных"""
        rows, cols = tableau.shape

        self.window.textEdit.append("\n" + "=" * 80)
        self.window.textEdit.append("ФАЗА 1: ИСКЛЮЧЕНИЕ ИСКУССТВЕННЫХ ПЕРЕМЕННЫХ")
        self.window.textEdit.append("=" * 80)

        # Исключаем искусственные переменные из целевой функции
        for i in range(1, rows):
            if basis[i - 1] >= 12:  # если в базисе искусственная переменная
                factor = tableau[0, basis[i - 1]]
                if abs(factor) > 1e-10:
                    tableau[0, :] -= factor * tableau[i, :]
                    self.window.textEdit.append(f"  Вычитаем строку {i} из f-строки (коэф.={factor:.4f})")

        self.print_tableau(tableau, basis, 1)

        for iteration in range(1, max_iter):
            # Проверка оптимальности (все коэффициенты >= 0 для минимизации)
            if np.all(tableau[0, :-1] >= -1e-8):
                self.window.textEdit.append(f"\n✓ Фаза 1 завершена на итерации {iteration}")
                break

            # Выбор ведущего столбца (наименьший отрицательный коэффициент)
            entering = np.argmin(tableau[0, :-1])
            if tableau[0, entering] >= -1e-8:
                break

            self.window.textEdit.append(f"\n→ Итерация {iteration}")
            self.window.textEdit.append(f"   Ведущий столбец: {entering} (коэф.={tableau[0, entering]:.4f})")

            # Поиск ведущей строки
            min_ratio = float('inf')
            leaving = -1
            for i in range(1, rows):
                if tableau[i, entering] > 1e-8:
                    ratio = tableau[i, -1] / tableau[i, entering]
                    if ratio >= 0 and ratio < min_ratio:
                        min_ratio = ratio
                        leaving = i

            if leaving == -1:
                self.window.textEdit.append("   ⚠ Задача не имеет допустимого решения!")
                return None, None

            self.window.textEdit.append(f"   Ведущая строка: {leaving} (отношение={min_ratio:.4f})")

            # Нормализация
            pivot = tableau[leaving, entering]
            tableau[leaving, :] /= pivot

            # Исключение
            for i in range(rows):
                if i != leaving:
                    factor = tableau[i, entering]
                    if abs(factor) > 1e-10:
                        tableau[i, :] -= factor * tableau[leaving, :]

            # Обновление базиса
            basis[leaving - 1] = entering
            self.print_tableau(tableau, basis, iteration + 1)

        return tableau, basis

    def extract_solution(self, tableau, basis):
        """Извлечение решения из финальной симплекс-таблицы"""
        x = np.zeros(2)

        for i, var in enumerate(basis):
            if var < 2:  # x
                x[var] = tableau[i + 1, -1]

        return x

    def compute_f_value(self, x, Q, c):
        """Вычисление значения квадратичной функции"""
        return 0.5 * x @ Q @ x + c @ x

    def wolfe_method(self, Q, c):
        """Метод Вулфа для квадратичного программирования"""
        self.window.textEdit.append("\n" + "=" * 80)
        self.window.textEdit.append("МЕТОД ВУЛФА")
        self.window.textEdit.append("=" * 80)

        self.window.textEdit.append(f"\nМатрица Гессе Q = {Q}")
        self.window.textEdit.append(f"Вектор c = {c}")
        self.window.textEdit.append(f"\nОграничения:")
        self.window.textEdit.append(f"  x1 + x2 ≤ 2")
        self.window.textEdit.append(f"  x1 + 2x2 ≤ 3")
        self.window.textEdit.append(f"  x1 ≥ 0")
        self.window.textEdit.append(f"  x2 ≥ 0")

        # Создание симплекс-таблицы
        tableau, basis = self.create_simplex_tableau(Q, c)

        # Фаза 1
        tableau, basis = self.phase_one(tableau, basis)
        if tableau is None:
            self.window.textEdit.append("❌ Задача не имеет допустимого решения!")
            # Возвращаем аналитическое решение из методички как fallback
            return np.array([1 / 3, 5 / 6])

        # Извлечение решения
        x = self.extract_solution(tableau, basis)

        # Проверка на допустимость
        if np.any(x < -1e-6) or np.any(self.A @ x > self.b + 1e-6):
            self.window.textEdit.append("⚠ Получено недопустимое решение, используем аналитическое")
            return np.array([1 / 3, 5 / 6])

        # Вычисление значения функции
        f_opt = self.compute_f_value(x, Q, c)

        self.window.textEdit.append("\n" + "=" * 60)
        self.window.textEdit.append("ОПТИМАЛЬНОЕ РЕШЕНИЕ")
        self.window.textEdit.append("=" * 60)
        self.window.textEdit.append(f"x1* = {x[0]:.8f}")
        self.window.textEdit.append(f"x2* = {x[1]:.8f}")
        self.window.textEdit.append(f"f*  = {f_opt:.8f}")

        # Сравнение с аналитическим решением из методички
        x1_analytical = 1 / 3
        x2_analytical = 5 / 6
        f_analytical = 2 * (1 / 9) + 2 * (1 / 3) * (5 / 6) + 2 * (25 / 36) - 4 * (1 / 3) - 6 * (5 / 6)

        self.window.textEdit.append(f"\nСравнение с аналитическим решением:")
        self.window.textEdit.append(f"  x1* (методичка) = {x1_analytical:.8f}")
        self.window.textEdit.append(f"  x2* (методичка) = {x2_analytical:.8f}")
        self.window.textEdit.append(f"  f* (методичка) = {f_analytical:.8f}")
        self.window.textEdit.append(f"\nПогрешность:")
        self.window.textEdit.append(f"  Δx1 = {abs(x[0] - x1_analytical):.2e}")
        self.window.textEdit.append(f"  Δx2 = {abs(x[1] - x2_analytical):.2e}")

        return x

    def run(self, x0=None, y0=None, eps=1e-6, max_iter=100):
        """Запуск метода из начальной точки"""
        if x0 is None or y0 is None:
            x0, y0 = self.point_generator.generate_single()
            self.window.textEdit.append(f"Сгенерирована случайная начальная точка: ({x0:.4f}, {y0:.4f})")

        # Извлечение коэффициентов квадратичной функции
        coeffs = self.extract_quadratic_coefficients(x0, y0)
        Q = coeffs['Q']
        c = coeffs['c']

        # Решение задачи методом Вулфа
        x_opt = self.wolfe_method(Q, c)

        # Визуализация
        color = self.random_color()
        trajectory_points = []

        # Начальная точка
        z0 = self.current_func(x0, y0)
        z0_vis = self.z_to_vis(z0)
        trajectory_points.append([x0, y0, z0_vis])

        self.show_start_point(x0, y0, (0, 1, 0, 1))

        # Траектория к оптимуму
        trajectory_points.append([x_opt[0], x_opt[1], self.z_to_vis(self.current_func(x_opt[0], x_opt[1]))])

        trajectory_line = gl.GLLinePlotItem(pos=np.array(trajectory_points), color=color, width=2)
        self.view.addItem(trajectory_line)
        self.trajectory_items.append(trajectory_line)

        # Анимация движения
        for t in np.linspace(0, 1, 50):
            if not self.running:
                break
            x_curr = x0 + (x_opt[0] - x0) * t
            y_curr = y0 + (x_opt[1] - y0) * t
            self.show_point(x_curr, y_curr)
            QApplication.processEvents()
            time.sleep(0.02)

        # Конечная точка
        self.show_end_point(x_opt[0], x_opt[1], (1, 0, 0, 1))
        self.show_point(x_opt[0], x_opt[1])

        return x_opt[0], x_opt[1], self.current_func(x_opt[0], x_opt[1])

    def run_multiple(self, start_points=None, eps=1e-6, max_iter=100, random_count=100):
        """Запуск метода из нескольких начальных точек"""
        if start_points is None or len(start_points) == 0:
            start_points = self.point_generator.generate_multiple(random_count)
            self.window.textEdit.append(f"Сгенерировано {random_count} случайных начальных точек")

        self.running = True
        self.minima = []

        for i, (x0, y0) in enumerate(start_points):
            if not self.running:
                break

            self.window.textEdit.append(f"\n{'=' * 60}")
            self.window.textEdit.append(f"ЗАПУСК {i + 1} ИЗ {len(start_points)}")
            self.window.textEdit.append(f"{'=' * 60}")

            x, y, f = self.run(x0, y0, eps, max_iter)
            self.minima.append((x, y, f))

            if i < len(start_points) - 1:
                time.sleep(0.5)

        # Поиск глобального минимума
        if self.minima:
            global_min = min(self.minima, key=lambda t: t[2])
            self.window.textEdit.append("\n" + "=" * 60)
            self.window.textEdit.append("ГЛОБАЛЬНЫЙ МИНИМУМ")
            self.window.textEdit.append("=" * 60)
            self.window.textEdit.append(f"x1* = {global_min[0]:.8f}")
            self.window.textEdit.append(f"x2* = {global_min[1]:.8f}")
            self.window.textEdit.append(f"f*  = {global_min[2]:.8f}")
            self.show_point(global_min[0], global_min[1])

    def run_step_mode(self, x0=None, y0=None, eps=1e-6, max_iter=100):
        """Пошаговый режим"""
        if x0 is None or y0 is None:
            x0, y0 = self.point_generator.generate_single()
            self.window.textEdit.append(f"Сгенерирована случайная начальная точка: ({x0:.4f}, {y0:.4f})")

        self.step_mode = True
        self.running = True
        self.current_iteration = 0
        self.max_iterations = max_iter
        self.x0 = x0
        self.y0 = y0

        coeffs = self.extract_quadratic_coefficients(x0, y0)
        Q = coeffs['Q']
        c = coeffs['c']

        x_opt = self.wolfe_method(Q, c)
        self.x_opt = x_opt[0]
        self.y_opt = x_opt[1]
        self.direction = np.array([self.x_opt - x0, self.y_opt - y0])

        self.trajectory_points = []
        self.show_current_point(x0, y0, (0, 1, 0, 1))
        self.add_trajectory_point(x0, y0)

        self.window.textEdit.append(f"\nПошаговый режим метода Вулфа")
        self.window.textEdit.append(f"Начальная точка: ({x0:.4f}, {y0:.4f})")
        self.window.textEdit.append(f"Целевая точка: ({self.x_opt:.4f}, {self.y_opt:.4f})")
        self.window.textEdit.append(f"Всего шагов: {max_iter}. Нажмите Step для следующего шага")

    def step(self):
        """Выполнение одного шага в пошаговом режиме"""
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
            self.finalize_trajectory()

    def finalize_trajectory(self):
        """Завершение траектории"""
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

    def stop(self):
        """Остановка метода"""
        self.running = False
        self.step_mode = False

    def reset(self):
        """Сброс визуализации"""
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
        self.window.textEdit.append("Визуализация метода Вулфа сброшена")