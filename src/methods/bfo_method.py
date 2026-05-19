# src/methods/bfo_method.py
import numpy as np
from PySide6.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import time


class BFOMethod:
    """
    Алгоритм бактериальной оптимизации (Bacterial Foraging Optimization)
    для максимизации функции гиперсферы: f(x,y) = -(x² + y²)
    Максимум в точке (0,0), значение 0
    """

    def __init__(self, view, current_func, current_zmin, current_zmax, point_item, window):
        self.view = view
        self.current_func = current_func
        self.current_zmin = current_zmin
        self.current_zmax = current_zmax
        self.point_item = point_item
        self.window = window

        # Состояние алгоритма
        self.running = False
        self.step_mode = False
        self.current_step = 0

        # Параметры алгоритма (из методички)
        self.n_bacteria = 20  # |S| - количество бактерий (чётное)
        self.n_chemotaxis_steps = 50  # ħ - шагов хемотаксиса
        self.n_reproduction_steps = 4  # ħ' - шагов репродукции
        self.n_elimination_steps = 2  # ħ'' - шагов ликвидации
        self.step_size = 0.1  # λ - величина шага хемотаксиса
        self.elimination_prob = 0.25  # ξe - вероятность ликвидации

        # Границы поиска
        self.xmin = -5
        self.xmax = 5
        self.ymin = -5
        self.ymax = 5

        # Популяция бактерий
        self.bacteria = []  # позиции бактерий
        self.health = []  # состояние здоровья бактерий

        # Лучшее решение
        self.best_position = None
        self.best_value = -np.inf

        # Визуализация
        self.bacteria_item = None
        self.best_item = None
        self.temp_point_item = None

    def set_function(self, func, zmin, zmax):
        """Установка целевой функции"""
        self.current_func = func
        self.current_zmin = zmin
        self.current_zmax = zmax

    def update_bounds(self, xmin, xmax, ymin, ymax):
        """Обновление границ поиска"""
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def set_parameters(self, n_bacteria, chemotaxis_steps, reproduction_steps,
                       elimination_steps, step_size, elimination_prob):
        """Установка параметров алгоритма из UI"""
        self.n_bacteria = n_bacteria
        self.n_chemotaxis_steps = chemotaxis_steps
        self.n_reproduction_steps = reproduction_steps
        self.n_elimination_steps = elimination_steps
        self.step_size = step_size
        self.elimination_prob = elimination_prob

    def z_to_vis(self, z):
        """Преобразование значения функции в визуальную высоту"""
        if self.current_zmax == self.current_zmin:
            return self.current_zmax
        return (z - self.current_zmin) / (self.current_zmax - self.current_zmin) * 10

    def objective(self, x, y):
        """Целевая функция (гиперсфера)"""
        return self.current_func(x, y)

    def fitness(self, x, y):
        """Фитнес-функция (значение целевой функции)"""
        return self.objective(x, y)

    def compute_health(self, trajectory_values):
        """
        Вычисление состояния здоровья бактерии
        h = сумма значений фитнес-функции во всех точках траектории
        """
        return sum(trajectory_values)

    def normalize_direction(self, dx, dy):
        """Нормирование направления (получение единичного вектора)"""
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm < 1e-10:
            return 0, 0
        return dx / norm, dy / norm

    def show_point(self, x, y):
        """Отображение текущей точки на графике"""
        if self.current_func is None:
            return
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])
        if self.temp_point_item:
            self.view.removeItem(self.temp_point_item)
        self.temp_point_item = gl.GLScatterPlotItem(
            pos=pos,
            size=2,  # было 8 → стало 2 (в 4 раза меньше)
            color=(1, 0, 0, 1)
        )
        self.view.addItem(self.temp_point_item)

    def update_visualization(self):
        """Обновление визуализации бактерий"""
        if self.bacteria_item:
            self.view.removeItem(self.bacteria_item)
        if self.best_item:
            self.view.removeItem(self.best_item)

        # Визуализация всех бактерий (синие) - размер уменьшен в 5 раз
        if len(self.bacteria) > 0:
            positions = []
            for (x, y) in self.bacteria:
                z = self.objective(x, y)
                z_vis = self.z_to_vis(z)
                positions.append([x, y, z_vis])

            self.bacteria_item = gl.GLScatterPlotItem(
                pos=np.array(positions),
                color=(0.3, 0.3, 1.0, 0.7),
                size=0.6,  # было 3 → стало 0.6 (в 5 раз меньше)
                pxMode=False
            )
            self.view.addItem(self.bacteria_item)

        # Визуализация лучшей бактерии (жёлтая) - размер уменьшен в 5 раз
        if self.best_position is not None:
            x, y = self.best_position
            z = self.objective(x, y)
            z_vis = self.z_to_vis(z)
            self.best_item = gl.GLScatterPlotItem(
                pos=np.array([[x, y, z_vis]]),
                size=2,  # было 10 → стало 2 (в 5 раз меньше)
                color=(1, 0.8, 0, 1),
                pxMode=False
            )
            self.view.addItem(self.best_item)
    def initialize_population(self):
        """Инициализация популяции бактерий в случайных точках"""
        self.bacteria = []
        for _ in range(self.n_bacteria):
            x = np.random.uniform(self.xmin, self.xmax)
            y = np.random.uniform(self.ymin, self.ymax)
            self.bacteria.append((x, y))

        self.update_best()
        self.window.textEdit.append(f"Инициализация: {self.n_bacteria} бактерий")
        self.window.textEdit.append(f"Границы: x∈[{self.xmin},{self.xmax}], y∈[{self.ymin},{self.ymax}]")

    def update_best(self):
        """Обновление лучшего решения"""
        for x, y in self.bacteria:
            value = self.objective(x, y)
            if value > self.best_value:
                self.best_value = value
                self.best_position = (x, y)

    def chemotaxis_step(self, x, y):
        """
        Один шаг хемотаксиса
        Формула: X' = X + λ * (V / ||V||)
        где V — случайное направление, λ — величина шага
        """
        # Случайное направление
        angle = np.random.uniform(0, 2 * np.pi)
        vx = np.cos(angle)
        vy = np.sin(angle)

        # Нормированное направление (уже единичное, т.к. cos²+sin²=1)
        # Движение
        new_x = x + self.step_size * vx
        new_y = y + self.step_size * vy

        # Ограничение границ
        new_x = np.clip(new_x, self.xmin, self.xmax)
        new_y = np.clip(new_y, self.ymin, self.ymax)

        return new_x, new_y

    def chemotaxis(self):
        """
        Процедура хемотаксиса — локальный поиск
        Каждая бактерия делает ħ шагов, запоминая лучшую позицию
        """
        for i in range(self.n_bacteria):
            x, y = self.bacteria[i]
            best_x, best_y = x, y
            best_fitness = self.fitness(x, y)

            # Запоминаем траекторию для вычисления здоровья
            trajectory = [best_fitness]

            for _ in range(self.n_chemotaxis_steps):
                if not self.running:
                    return

                # Пробуем сделать шаг
                new_x, new_y = self.chemotaxis_step(best_x, best_y)
                new_fitness = self.fitness(new_x, new_y)

                # Если стало лучше — сохраняем
                if new_fitness > best_fitness:
                    best_x, best_y = new_x, new_y
                    best_fitness = new_fitness

                trajectory.append(best_fitness)

            # Сохраняем лучшую позицию после хемотаксиса
            self.bacteria[i] = (best_x, best_y)
            # Сохраняем здоровье (сумма всех значений по траектории)
            self.health[i] = sum(trajectory)

    def reproduction(self):
        """
        Процедура репродукции
        Половина лучших бактерий выживает, каждая расщепляется на две
        Половина худших погибает
        """
        # Сортировка бактерий по здоровью (убывание)
        sorted_indices = np.argsort(self.health)[::-1]

        new_bacteria = []
        half = self.n_bacteria // 2

        # Выжившие (первые half) расщепляются на две
        for j in range(half):
            idx = sorted_indices[j]
            x, y = self.bacteria[idx]
            new_bacteria.append((x, y))
            new_bacteria.append((x, y))

        self.bacteria = new_bacteria

        # Обновление здоровья
        for i in range(self.n_bacteria):
            x, y = self.bacteria[i]
            self.health[i] = self.fitness(x, y)

        self.window.textEdit.append(f"  Репродукция: выжило {half} бактерий")

    def elimination_dispersal(self):
        """
        Процедура ликвидации и рассеивания
        Некоторые бактерии с вероятностью ξe уничтожаются и заменяются новыми случайными
        """
        n_eliminated = 0
        for i in range(self.n_bacteria):
            if np.random.random() < self.elimination_prob:
                # Уничтожаем бактерию и создаём новую в случайной точке
                x = np.random.uniform(self.xmin, self.xmax)
                y = np.random.uniform(self.ymin, self.ymax)
                self.bacteria[i] = (x, y)
                self.health[i] = self.fitness(x, y)
                n_eliminated += 1

        if n_eliminated > 0:
            self.window.textEdit.append(f"  Ликвидация: уничтожено {n_eliminated} бактерий")

    def run(self):
        """Основной цикл алгоритма BFO"""
        if self.current_func is None:
            self.window.textEdit.append("Сначала постройте поверхность")
            return

        self.running = True
        self.best_value = -np.inf
        self.best_position = None
        self.health = [0] * self.n_bacteria

        # Вывод параметров
        self.window.textEdit.append("=" * 60)
        self.window.textEdit.append("АЛГОРИТМ БАКТЕРИАЛЬНОЙ ОПТИМИЗАЦИИ (BFO)")
        self.window.textEdit.append("=" * 60)
        self.window.textEdit.append(f"Количество бактерий |S| = {self.n_bacteria}")
        self.window.textEdit.append(f"Шагов хемотаксиса ħ = {self.n_chemotaxis_steps}")
        self.window.textEdit.append(f"Шагов репродукции ħ' = {self.n_reproduction_steps}")
        self.window.textEdit.append(f"Шагов ликвидации ħ'' = {self.n_elimination_steps}")
        self.window.textEdit.append(f"Величина шага λ = {self.step_size}")
        self.window.textEdit.append(f"Вероятность ликвидации ξe = {self.elimination_prob}")
        self.window.textEdit.append("-" * 60)

        # Инициализация
        self.initialize_population()
        self.update_visualization()
        QApplication.processEvents()

        iteration = 0
        total_iterations = self.n_elimination_steps * self.n_reproduction_steps * self.n_chemotaxis_steps

        # Основные циклы алгоритма
        for l in range(self.n_elimination_steps):  # Ликвидация и рассеивание
            if not self.running:
                break

            for r in range(self.n_reproduction_steps):  # Репродукция
                if not self.running:
                    break

                for t in range(self.n_chemotaxis_steps):  # Хемотаксис
                    if not self.running:
                        break

                    self.current_step = iteration

                    # 4.1 Хемотаксис
                    self.chemotaxis()

                    # Обновление лучшего решения
                    old_best = self.best_value
                    self.update_best()

                    # Вывод информации
                    if self.best_value > old_best + 1e-10:
                        self.window.textEdit.append(
                            f"Итерация {iteration + 1}: НОВЫЙ МАКСИМУМ f = {self.best_value:.8f} "
                            f"в точке ({self.best_position[0]:.4f}, {self.best_position[1]:.4f})"
                        )
                    elif iteration % 20 == 0:
                        self.window.textEdit.append(
                            f"Итерация {iteration + 1}: лучшее значение = {self.best_value:.8f}"
                        )

                    # Визуализация
                    self.update_visualization()
                    if self.best_position:
                        self.show_point(self.best_position[0], self.best_position[1])

                    QApplication.processEvents()
                    time.sleep(0.02)

                    iteration += 1

                # Репродукция (после каждого цикла хемотаксиса)
                if self.running:
                    self.reproduction()
                    self.update_visualization()
                    QApplication.processEvents()
                    time.sleep(0.05)

            # Ликвидация и рассеивание (после каждого цикла репродукции)
            if self.running:
                self.elimination_dispersal()
                self.update_visualization()
                QApplication.processEvents()
                time.sleep(0.05)

        # Результаты
        self.window.textEdit.append("-" * 60)
        self.window.textEdit.append("РЕЗУЛЬТАТ ОПТИМИЗАЦИИ")
        if self.best_position:
            self.window.textEdit.append(f"x* = {self.best_position[0]:.8f}")
            self.window.textEdit.append(f"y* = {self.best_position[1]:.8f}")
            self.window.textEdit.append(f"f(x*,y*) = {self.best_value:.8f}")

        return self.best_position[0] if self.best_position else 0, \
            self.best_position[1] if self.best_position else 0, \
            self.best_value

    def stop(self):
        """Остановка алгоритма"""
        self.running = False

    def reset(self):
        """Сброс визуализации"""
        self.running = False
        self.bacteria = []
        self.best_position = None
        self.best_value = -np.inf

        if self.bacteria_item:
            self.view.removeItem(self.bacteria_item)
            self.bacteria_item = None
        if self.best_item:
            self.view.removeItem(self.best_item)
            self.best_item = None
        if self.temp_point_item:
            self.view.removeItem(self.temp_point_item)
            self.temp_point_item = None

        self.window.textEdit.append("Визуализация BFO сброшена")