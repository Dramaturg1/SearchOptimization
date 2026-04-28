# src/methods/bees_method.py
import numpy as np
from PySide6.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import time


class BeesMethod:
    def __init__(self, view, current_func, current_zmin, current_zmax, window):
        self.view = view
        self.current_func = current_func
        self.current_zmin = current_zmin
        self.current_zmax = current_zmax
        self.window = window

        self.running = False
        self.step_mode = False
        self.current_iteration = 0
        self.max_iterations = 0

        # Параметры алгоритма (из методички, стр. 6)
        self.n_scouts = 16  # S° – число пчел-разведчиков
        self.n_elite_sites = 2  # A^b – число элитных участков
        self.n_perspective_sites = 3  # A^p – число перспективных участков
        self.n_elite_bees = 7  # n^b – число пчел на элитных участках
        self.n_perspective_bees = 4  # n^p – число пчел на перспективных участках
        self.radius = 0.2  # r – радиус участков
        self.stagnation_limit = 20  # δ – лимит стагнации для остановки

        # Состояние алгоритма
        self.scouts = []  # позиции пчел-разведчиков
        self.scout_values = []  # значения в этих точках
        self.best_position = None
        self.best_value = np.inf
        self.best_positions_history = []

        # Визуализация
        self.scouts_item = None
        self.worker_bees_item = None
        self.best_bee_item = None
        self.temp_point_item = None

        # Границы
        self.xmin = -5.12
        self.xmax = 5.12
        self.ymin = -5.12
        self.ymax = 5.12

        # Счетчик стагнации
        self.stagnation_counter = 0

    def set_function(self, func, zmin, zmax):
        self.current_func = func
        self.current_zmin = zmin
        self.current_zmax = zmax

    def update_bounds(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def z_to_vis(self, z):
        if self.current_zmax == self.current_zmin:
            return self.current_zmax
        return (z - self.current_zmin) / (self.current_zmax - self.current_zmin) * 10

    def fitness(self, x, y):
        """Фитнес-функция (обратна целевой, т.к. ищем максимум)"""
        f = self.current_func(x, y)
        # Для минимизации используем обратную величину
        if f != 0:
            return 1.0 / (abs(f) + 1e-10)
        return 1e10

    def objective(self, x, y):
        """Целевая функция (для вывода результата)"""
        return self.current_func(x, y)

    def show_point(self, x, y):
        if self.current_func is None:
            return
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])
        if self.temp_point_item:
            self.view.removeItem(self.temp_point_item)
        self.temp_point_item = gl.GLScatterPlotItem(pos=pos, size=15, color=(1, 0, 0, 1))
        self.view.addItem(self.temp_point_item)

    def update_visualization(self):
        """Обновление визуализации всех пчел"""
        # Удаляем старые элементы
        if self.scouts_item:
            self.view.removeItem(self.scouts_item)
        if self.worker_bees_item:
            self.view.removeItem(self.worker_bees_item)
        if self.best_bee_item:
            self.view.removeItem(self.best_bee_item)

        # Отображаем пчел-разведчиков (синие) - размер уменьшен в 3 раза
        if len(self.scouts) > 0:
            positions = []
            for x, y in self.scouts:
                z = self.current_func(x, y)
                z_vis = self.z_to_vis(z)
                positions.append([x, y, z_vis])

            self.scouts_item = gl.GLScatterPlotItem(
                pos=np.array(positions),
                color=(0.3, 0.3, 1.0, 0.7),
                size=1,  # было 5 → стало 2 (уменьшено ~ в 2.5 раза)
                pxMode=False
            )
            self.view.addItem(self.scouts_item)

        # Отображаем лучшую пчелу (жёлтая) - размер уменьшен в 3 раза
        if self.best_position is not None:
            x, y = self.best_position[0], self.best_position[1]
            z = self.current_func(x, y)
            z_vis = self.z_to_vis(z)
            self.best_bee_item = gl.GLScatterPlotItem(
                pos=np.array([[x, y, z_vis]]),
                size=1,  # было 15 → стало 5 (уменьшено в 3 раза)
                color=(1, 0.8, 0, 1),
                pxMode=False
            )
            self.view.addItem(self.best_bee_item)

    def initialize_scouts(self):
        """Инициализация пчел-разведчиков в случайных точках"""
        self.scouts = []
        self.scout_values = []

        for _ in range(self.n_scouts):
            x = np.random.uniform(self.xmin, self.xmax)
            y = np.random.uniform(self.ymin, self.ymax)
            self.scouts.append((x, y))
            self.scout_values.append(self.fitness(x, y))

        # Находим лучшую точку
        best_idx = np.argmax(self.scout_values)
        best_x, best_y = self.scouts[best_idx]
        self.best_position = [best_x, best_y]  # ← список вместо кортежа
        self.best_value = self.objective(best_x, best_y)

    def local_search(self, center_x, center_y, radius, n_bees):
        """
        Локальный поиск на участке
        center_x, center_y – центр участка
        radius – радиус участка
        n_bees – количество пчел, отправляемых на участок
        Возвращает лучшую найденную точку и её фитнес
        """
        best_local = [center_x, center_y]  # ← список вместо кортежа
        best_local_fitness = self.fitness(center_x, center_y)
        bees_positions = []

        for _ in range(n_bees):
            # Случайная точка в пределах участка
            dx = np.random.uniform(-radius, radius)
            dy = np.random.uniform(-radius, radius)
            x = np.clip(center_x + dx, self.xmin, self.xmax)
            y = np.clip(center_y + dy, self.ymin, self.ymax)

            bees_positions.append((x, y))
            fit = self.fitness(x, y)

            if fit > best_local_fitness:
                best_local_fitness = fit
                best_local = [x, y]  # ← список вместо кортежа

        return best_local, best_local_fitness, bees_positions

    def run(self, max_iter=500, n_scouts=16):
        """Запуск пчелиного алгоритма"""
        self.n_scouts = n_scouts  # ← устанавливаем из параметра

        self.initialize_scouts()
        self.current_iteration = 0
        self.max_iterations = max_iter
        self.running = True
        self.stagnation_counter = 0

        prev_best_value = self.best_value

        self.window.textEdit.append(f"\n{'=' * 60}")
        self.window.textEdit.append("ЗАПУСК ПЧЕЛИНОГО АЛГОРИТМА (Bees Algorithm)")
        self.window.textEdit.append(f"{'=' * 60}")
        self.window.textEdit.append(f"Параметры (из методички):")
        self.window.textEdit.append(f"  Пчел-разведчиков (S°) = {self.n_scouts}")
        self.window.textEdit.append(f"  Элитных участков (A^b) = {self.n_elite_sites}")
        self.window.textEdit.append(f"  Перспективных участков (A^p) = {self.n_perspective_sites}")
        self.window.textEdit.append(f"  Пчел на элитных участках (n^b) = {self.n_elite_bees}")
        self.window.textEdit.append(f"  Пчел на перспективных (n^p) = {self.n_perspective_bees}")
        self.window.textEdit.append(f"  Радиус участков (r) = {self.radius}")
        self.window.textEdit.append(f"  Лимит стагнации (δ) = {self.stagnation_limit}")
        self.window.textEdit.append(f"Границы: x∈[{self.xmin},{self.xmax}], y∈[{self.ymin},{self.ymax}]")

        self.update_visualization()
        QApplication.processEvents()

        for iteration in range(max_iter):
            if not self.running:
                break

            self.current_iteration = iteration

            # 1. Сортировка участков по фитнесу
            sites = list(zip(self.scouts, self.scout_values))
            sites.sort(key=lambda s: s[1], reverse=True)

            # 2. Выделение элитных и перспективных участков
            elite_sites = sites[:self.n_elite_sites]
            perspective_sites = sites[self.n_elite_sites:self.n_elite_sites + self.n_perspective_sites]

            new_scouts = []
            new_scout_values = []

            # 3. Локальный поиск на элитных участках
            for (x, y), fit in elite_sites:
                best_local, best_local_fit, bees = self.local_search(
                    x, y, self.radius, self.n_elite_bees
                )
                new_scouts.append(best_local)
                new_scout_values.append(best_local_fit)

            # 4. Локальный поиск на перспективных участках
            for (x, y), fit in perspective_sites:
                best_local, best_local_fit, bees = self.local_search(
                    x, y, self.radius, self.n_perspective_bees
                )
                new_scouts.append(best_local)
                new_scout_values.append(best_local_fit)

            # 5. Остальные пчелы-разведчики (случайный поиск)
            remaining_scouts = self.n_scouts - len(new_scouts)
            for _ in range(remaining_scouts):
                x = np.random.uniform(self.xmin, self.xmax)
                y = np.random.uniform(self.ymin, self.ymax)
                new_scouts.append((x, y))
                new_scout_values.append(self.fitness(x, y))

            # 6. Обновляем рой
            self.scouts = new_scouts
            self.scout_values = new_scout_values

            # 7. Находим лучшее решение
            best_idx = np.argmax(self.scout_values)
            best_x, best_y = self.scouts[best_idx]
            current_best_pos = [best_x, best_y]  # ← список
            current_best_val = self.objective(best_x, best_y)

            # Проверка улучшения
            if current_best_val < self.best_value - 1e-10:
                self.best_value = current_best_val
                self.best_position = [best_x, best_y]  # ← список
                self.stagnation_counter = 0
                self.window.textEdit.append(f"Итерация {iteration + 1}: НОВЫЙ МИНИМУМ = {self.best_value:.8f}")
            else:
                self.stagnation_counter += 1
                if iteration % 10 == 0:
                    self.window.textEdit.append(f"Итерация {iteration + 1}: лучшее значение = {self.best_value:.8f}")
            # Обновляем визуализацию
            self.update_visualization()
            self.show_point(self.best_position[0], self.best_position[1])

            QApplication.processEvents()
            time.sleep(0.02)

            # Проверка стагнации
            if self.stagnation_counter >= self.stagnation_limit:
                self.window.textEdit.append(f"\nСтагнация в течение {self.stagnation_limit} итераций. Остановка.")
                break

        self.window.textEdit.append(f"\n{'=' * 60}")
        self.window.textEdit.append("РЕЗУЛЬТАТ ОПТИМИЗАЦИИ")
        self.window.textEdit.append(f"{'=' * 60}")
        self.window.textEdit.append(f"x* = {self.best_position[0]:.8f}")
        self.window.textEdit.append(f"y* = {self.best_position[1]:.8f}")
        self.window.textEdit.append(f"f(x*,y*) = {self.best_value:.8f}")

        return self.best_position[0], self.best_position[1], self.best_value

    def stop(self):
        self.running = False
        self.step_mode = False

    def reset(self):
        self.running = False
        self.step_mode = False
        self.current_iteration = 0
        self.scouts = []
        self.scout_values = []
        self.best_position = None
        self.best_value = np.inf

        if self.scouts_item:
            self.view.removeItem(self.scouts_item)
            self.scouts_item = None
        if self.worker_bees_item:
            self.view.removeItem(self.worker_bees_item)
            self.worker_bees_item = None
        if self.best_bee_item:
            self.view.removeItem(self.best_bee_item)
            self.best_bee_item = None
        if self.temp_point_item:
            self.view.removeItem(self.temp_point_item)
            self.temp_point_item = None

        self.window.textEdit.append("Визуализация пчелиного алгоритма сброшена")