# src/methods/ais_method.py
import numpy as np
from PySide6.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import time
import random


class AISMethod:

    def __init__(self, view, current_func, current_zmin, current_zmax, point_item, window):
        self.view = view
        self.current_func = current_func
        self.current_zmin = current_zmin
        self.current_zmax = current_zmax
        self.point_item = point_item
        self.window = window

        self.running = False
        self.step_mode = False
        self.current_iteration = 0
        self.max_iterations = 0

        self.n_antibodies = 50
        self.n_best = 10
        self.n_random = 5
        self.n_clones = 10
        self.mutation_rate = 0.5
        self.n_keep = 5

        self.affinity_threshold = 0.1
        self.network_threshold = 0.05

        self.xmin = -5
        self.xmax = 5
        self.ymin = -5
        self.ymax = 5

        self.antibodies = []
        self.memory_cells = []
        self.best_antibody = None
        self.best_value = np.inf
        self.best_history = []

        self.antibodies_item = None
        self.memory_cells_item = None
        self.best_antibody_item = None
        self.temp_point_item = None
        self.trajectory_points = []
        self.trajectory_line = None

    def set_function(self, func, zmin, zmax):
        self.current_func = func
        self.current_zmin = zmin
        self.current_zmax = zmax

    def update_bounds(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def set_parameters(self, n_antibodies, max_iterations, n_best, n_random, n_clones, mutation_rate):
        self.n_antibodies = n_antibodies
        self.max_iterations = max_iterations
        self.n_best = n_best
        self.n_random = n_random
        self.n_clones = n_clones
        self.mutation_rate = mutation_rate
        self.n_keep = max(1, n_best // 2)

    def z_to_vis(self, z):
        if self.current_zmax == self.current_zmin:
            return self.current_zmax
        return (z - self.current_zmin) / (self.current_zmax - self.current_zmin) * 10

    def objective(self, x, y):
        return self.current_func(x, y)

    def compute_affinity(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def compute_bg_affinity(self, antibody, antigen):
        x, y = antibody
        value = self.objective(x, y)
        if abs(value) < 1e-10:
            return 1.0
        return 1.0 / (1.0 + abs(value))

    def show_point(self, x, y):
        if self.current_func is None:
            return
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])
        if self.temp_point_item:
            self.view.removeItem(self.temp_point_item)
        self.temp_point_item = gl.GLScatterPlotItem(pos=pos, size=4, color=(1, 0, 0, 1))
        self.view.addItem(self.temp_point_item)

    def update_visualization(self):
        if self.antibodies_item:
            self.view.removeItem(self.antibodies_item)
        if self.memory_cells_item:
            self.view.removeItem(self.memory_cells_item)
        if self.best_antibody_item:
            self.view.removeItem(self.best_antibody_item)

        if len(self.antibodies) > 0:
            positions = []
            colors = []
            for i, (x, y) in enumerate(self.antibodies):
                z = self.objective(x, y)
                z_vis = self.z_to_vis(z)
                positions.append([x, y, z_vis])
                affinity = self.compute_bg_affinity((x, y), None)
                colors.append([1 - affinity, affinity * 0.5, affinity, 0.6])

            self.antibodies_item = gl.GLScatterPlotItem(
                pos=np.array(positions),
                color=np.array(colors),
                size=0.3,
                pxMode=False
            )
            self.view.addItem(self.antibodies_item)

        if len(self.memory_cells) > 0:
            positions = []
            for (x, y) in self.memory_cells:
                z = self.objective(x, y)
                z_vis = self.z_to_vis(z)
                positions.append([x, y, z_vis])

            self.memory_cells_item = gl.GLScatterPlotItem(
                pos=np.array(positions),
                color=(1, 0.8, 0, 0.8),
                size=0.3,
                pxMode=False
            )
            self.view.addItem(self.memory_cells_item)

        if self.best_antibody is not None:
            x, y = self.best_antibody
            z = self.objective(x, y)
            z_vis = self.z_to_vis(z)
            self.best_antibody_item = gl.GLScatterPlotItem(
                pos=np.array([[x, y, z_vis]]),
                color=(1, 0, 0, 1),
                size=0.5,
                pxMode=False
            )
            self.view.addItem(self.best_antibody_item)

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
                width=2
            )
            self.view.addItem(self.trajectory_line)

    def initialize_population(self):
        self.antibodies = []
        self.memory_cells = []

        for _ in range(self.n_antibodies):
            x = np.random.uniform(self.xmin, self.xmax)
            y = np.random.uniform(self.ymin, self.ymax)
            self.antibodies.append((x, y))

        self.update_best()

        self.window.textEdit.append(f"Популяция инициализирована: {self.n_antibodies} антител")
        self.window.textEdit.append(f"Границы поиска: x∈[{self.xmin}, {self.xmax}], y∈[{self.ymin}, {self.ymax}]")

    def update_best(self):
        for x, y in self.antibodies:
            value = self.objective(x, y)
            if value < self.best_value:
                self.best_value = value
                self.best_antibody = (x, y)

    def select_best_antibodies(self):
        affinities = []
        for i, (x, y) in enumerate(self.antibodies):
            affinity = self.compute_bg_affinity((x, y), None)
            affinities.append((i, affinity))

        affinities.sort(key=lambda x: x[1], reverse=True)

        n_select = min(self.n_best, len(self.antibodies))
        best_indices = [affinities[i][0] for i in range(n_select)]

        return [self.antibodies[i] for i in best_indices]

    def clone_antibodies(self, selected_antibodies):
        clones = []

        for antibody in selected_antibodies:
            affinity = self.compute_bg_affinity(antibody, None)
            n_clones_for_this = max(1, int(self.n_clones * affinity))

            for _ in range(n_clones_for_this):
                clones.append(list(antibody))

        return clones

    def mutate_clones(self, clones):
        mutated = []

        for clone in clones:
            affinity = self.compute_bg_affinity(tuple(clone), None)
            adaptive_rate = self.mutation_rate * np.exp(-affinity * 5)

            dx = adaptive_rate * np.random.uniform(-0.5, 0.5) * (self.xmax - self.xmin)
            dy = adaptive_rate * np.random.uniform(-0.5, 0.5) * (self.ymax - self.ymin)

            clone[0] += dx
            clone[1] += dy

            clone[0] = np.clip(clone[0], self.xmin, self.xmax)
            clone[1] = np.clip(clone[1], self.ymin, self.ymax)

            mutated.append(tuple(clone))

        return mutated

    def select_best_clones(self, clones):
        if len(clones) == 0:
            return []

        clone_affinities = []
        for i, clone in enumerate(clones):
            affinity = self.compute_bg_affinity(clone, None)
            clone_affinities.append((i, affinity, clone))

        clone_affinities.sort(key=lambda x: x[1], reverse=True)

        n_select = min(self.n_keep, len(clones))
        best_clones = [clone_affinities[i][2] for i in range(n_select)]

        return best_clones

    def update_memory_cells(self, new_cells):
        self.memory_cells.extend(new_cells)

        self.memory_cells = list(set(self.memory_cells))

        filtered_memory = []
        for cell in self.memory_cells:
            affinity = self.compute_bg_affinity(cell, None)
            if affinity >= self.affinity_threshold:
                filtered_memory.append(cell)

        self.memory_cells = filtered_memory

        if len(self.memory_cells) > self.n_antibodies:
            affinities = []
            for i, cell in enumerate(self.memory_cells):
                aff = self.compute_bg_affinity(cell, None)
                affinities.append((i, aff, cell))
            affinities.sort(key=lambda x: x[1], reverse=True)
            self.memory_cells = [affinities[i][2] for i in range(self.n_antibodies)]

    def network_suppression(self):
        if len(self.antibodies) < 2:
            return

        to_remove = set()

        for i in range(len(self.antibodies)):
            for j in range(i + 1, len(self.antibodies)):
                x1, y1 = self.antibodies[i]
                x2, y2 = self.antibodies[j]
                distance = self.compute_affinity(x1, y1, x2, y2)

                if distance < self.network_threshold * (self.xmax - self.xmin):
                    aff1 = self.compute_bg_affinity(self.antibodies[i], None)
                    aff2 = self.compute_bg_affinity(self.antibodies[j], None)
                    if aff1 < aff2:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)

        new_antibodies = [self.antibodies[i] for i in range(len(self.antibodies)) if i not in to_remove]
        self.antibodies = new_antibodies

    def add_random_antibodies(self):
        if len(self.antibodies) < 2:
            return

        affinities = []
        for i, antibody in enumerate(self.antibodies):
            aff = self.compute_bg_affinity(antibody, None)
            affinities.append((i, aff, antibody))

        affinities.sort(key=lambda x: x[1])

        n_replace = min(self.n_random, len(self.antibodies) // 4)
        n_replace = max(1, n_replace)

        for i in range(n_replace):
            if i < len(affinities):
                idx = affinities[i][0]
                x = np.random.uniform(self.xmin, self.xmax)
                y = np.random.uniform(self.ymin, self.ymax)
                self.antibodies[idx] = (x, y)

    def run(self):
        if self.current_func is None:
            self.window.textEdit.append("Сначала постройте поверхность")
            return

        self.window.textEdit.append("=" * 50)
        self.window.textEdit.append("АЛГОРИТМ ИСКУССТВЕННОЙ ИММУННОЙ СЕТИ (AIS)")
        self.window.textEdit.append("=" * 50)
        self.window.textEdit.append(f"Размер популяции антител |S| = {self.n_antibodies}")
        self.window.textEdit.append(f"Число лучших антител для клонирования n_b = {self.n_best}")
        self.window.textEdit.append(f"Число случайных антител для замены b_n = {self.n_random}")
        self.window.textEdit.append(f"Число клонов на антитело n_c = {self.n_clones}")
        self.window.textEdit.append(f"Коэффициент мутации α = {self.mutation_rate}")
        self.window.textEdit.append(f"Максимум итераций = {self.max_iterations}")
        self.window.textEdit.append("-" * 50)

        self.running = True
        self.best_value = np.inf
        self.best_antibody = None
        self.trajectory_points = []

        self.initialize_population()
        self.update_visualization()
        QApplication.processEvents()

        for iteration in range(self.max_iterations):
            if not self.running:
                self.window.textEdit.append("\nАлгоритм остановлен пользователем")
                break

            self.current_iteration = iteration

            best_antibodies = self.select_best_antibodies()

            clones = self.clone_antibodies(best_antibodies)

            mutated_clones = self.mutate_clones(clones)

            best_clones = self.select_best_clones(mutated_clones)

            self.update_memory_cells(best_clones)

            self.antibodies.extend(best_clones)

            self.network_suppression()

            self.add_random_antibodies()

            old_best = self.best_value
            self.update_best()

            if iteration % (max(1, self.max_iterations // 20)) == 0 or self.best_value < old_best - 1e-10:
                self.window.textEdit.append(
                    f"Итерация {iteration + 1}/{self.max_iterations}: "
                    f"лучшее значение = {self.best_value:.8f}"
                )

            if self.best_antibody:
                self.add_trajectory_point(self.best_antibody[0], self.best_antibody[1])

            self.update_visualization()
            if self.best_antibody:
                self.show_point(self.best_antibody[0], self.best_antibody[1])

            QApplication.processEvents()
            time.sleep(0.02)

        # Результаты
        self.window.textEdit.append("-" * 50)
        self.window.textEdit.append("РЕЗУЛЬТАТ ОПТИМИЗАЦИИ")
        if self.best_antibody:
            self.window.textEdit.append(f"x* = {self.best_antibody[0]:.8f}")
            self.window.textEdit.append(f"y* = {self.best_antibody[1]:.8f}")
            self.window.textEdit.append(f"f(x*,y*) = {self.best_value:.8f}")

        return self.best_antibody[0] if self.best_antibody else 0, \
            self.best_antibody[1] if self.best_antibody else 0, \
            self.best_value

    def run_step_mode(self):
        if self.current_func is None:
            self.window.textEdit.append("Сначала постройте поверхность")
            return

        self.running = True
        self.step_mode = True
        self.current_iteration = 0
        self.best_value = np.inf
        self.best_antibody = None
        self.trajectory_points = []

        self.initialize_population()
        self.update_visualization()

        self.window.textEdit.append(f"\nПошаговый режим AIS")
        self.window.textEdit.append(f"Параметры: |S|={self.n_antibodies}, n_b={self.n_best}, "
                                    f"n_c={self.n_clones}, α={self.mutation_rate}")
        self.window.textEdit.append(f"Итераций: {self.max_iterations}")

    def step(self):
        if not self.step_mode or not self.running:
            return

        if self.current_iteration >= self.max_iterations:
            self.window.textEdit.append("Достигнуто максимальное число итераций")
            return

        best_antibodies = self.select_best_antibodies()

        clones = self.clone_antibodies(best_antibodies)

        mutated_clones = self.mutate_clones(clones)

        best_clones = self.select_best_clones(mutated_clones)

        self.update_memory_cells(best_clones)

        self.antibodies.extend(best_clones)

        self.network_suppression()

        self.add_random_antibodies()

        old_best = self.best_value
        self.update_best()

        if self.best_antibody:
            self.add_trajectory_point(self.best_antibody[0], self.best_antibody[1])

        self.update_visualization()
        if self.best_antibody:
            self.show_point(self.best_antibody[0], self.best_antibody[1])

        self.current_iteration += 1

        self.window.textEdit.append(
            f"Шаг {self.current_iteration}/{self.max_iterations}: "
            f"лучшее f = {self.best_value:.8f}"
        )

        QApplication.processEvents()

    def stop(self):
        self.running = False
        self.step_mode = False

    def reset(self):
        self.running = False
        self.step_mode = False
        self.current_iteration = 0
        self.antibodies = []
        self.memory_cells = []
        self.best_antibody = None
        self.best_value = np.inf
        self.trajectory_points = []

        if self.antibodies_item:
            self.view.removeItem(self.antibodies_item)
            self.antibodies_item = None
        if self.memory_cells_item:
            self.view.removeItem(self.memory_cells_item)
            self.memory_cells_item = None
        if self.best_antibody_item:
            self.view.removeItem(self.best_antibody_item)
            self.best_antibody_item = None
        if self.temp_point_item:
            self.view.removeItem(self.temp_point_item)
            self.temp_point_item = None
        if self.trajectory_line:
            self.view.removeItem(self.trajectory_line)
            self.trajectory_line = None

        self.window.textEdit.append("Визуализация иммунного алгоритма сброшена")