# src/methods/hybrid_ga_pso.py
import numpy as np
from PySide6.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import time
import random


class HybridGA_PSO:


    def __init__(self, view, current_func, current_zmin, current_zmax, point_item, window):
        self.view = view
        self.current_func = current_func
        self.current_zmin = current_zmin
        self.current_zmax = current_zmax
        self.point_item = point_item
        self.window = window

        self.running = False

        # Параметры GA (глобальный поиск)
        self.ga_population_size = 50
        self.ga_max_generations = 100
        self.ga_mutation_rate = 0.1
        self.ga_crossover_rate = 0.8
        self.ga_elite_size = 2

        # Параметры PSO (локальная оптимизация)
        self.pso_particles = 30
        self.pso_max_iterations = 50
        self.pso_velocity_ratio = 0.5
        self.pso_local_ratio = 2.0
        self.pso_global_ratio = 5.0

        # Границы поиска
        self.xmin = -5
        self.xmax = 5
        self.ymin = -5
        self.ymax = 5

        # Состояние
        self.current_phase = "GA"
        self.current_generation = 0
        self.best_solution = None
        self.best_value = np.inf
        self.best_history = []

        # Популяции
        self.ga_population = []
        self.pso_particles_list = []
        self.pso_global_best = None
        self.pso_global_best_value = np.inf

        # Визуализация
        self.population_item = None
        self.particles_item = None
        self.best_point_item = None
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

    def set_parameters(self, ga_pop_size, ga_generations, pso_particles, pso_iterations):
        self.ga_population_size = ga_pop_size
        self.ga_max_generations = ga_generations
        self.pso_particles = pso_particles
        self.pso_max_iterations = pso_iterations

    def z_to_vis(self, z):
        if self.current_zmax == self.current_zmin:
            return self.current_zmax
        return (z - self.current_zmin) / (self.current_zmax - self.current_zmin) * 10

    def objective(self, x, y):
        return self.current_func(x, y)

    def show_point(self, x, y):
        if self.current_func is None:
            return
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])
        if self.temp_point_item:
            self.view.removeItem(self.temp_point_item)
        self.temp_point_item = gl.GLScatterPlotItem(pos=pos, size=5, color=(1, 0, 0, 1))
        self.view.addItem(self.temp_point_item)

    def show_best_point(self, x, y):
        if self.current_func is None:
            return
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])

        if self.best_point_item:
            self.view.removeItem(self.best_point_item)

        self.best_point_item = gl.GLScatterPlotItem(
            pos=pos,
            color=(1, 0.8, 0, 1),
            size=3,
            pxMode=False
        )
        self.view.addItem(self.best_point_item)

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

    def show_ga_population(self, population):
        if len(population) == 0:
            return

        if self.population_item:
            self.view.removeItem(self.population_item)

        points = []
        colors = []

        f_values = []
        for x, y in population:
            z = self.current_func(x, y)
            f_values.append(z)

        f_min = min(f_values) if f_values else 0
        f_max = max(f_values) if f_values else 1

        for (x, y), z in zip(population, f_values):
            z_vis = self.z_to_vis(z)
            points.append([x, y, z_vis])

            if f_max > f_min:
                norm = (z - f_min) / (f_max - f_min)
            else:
                norm = 0
            colors.append([norm, 1 - norm, 0, 0.7])

        self.population_item = gl.GLScatterPlotItem(
            pos=np.array(points),
            color=np.array(colors),
            size=0.7,
            pxMode=False
        )
        self.view.addItem(self.population_item)

    def show_pso_particles(self, particles):
        if len(particles) == 0:
            return

        if self.particles_item:
            self.view.removeItem(self.particles_item)

        points = []
        for p in particles:
            x, y = p.current_position
            z = self.current_func(x, y)
            z_vis = self.z_to_vis(z)
            points.append([x, y, z_vis])

        self.particles_item = gl.GLScatterPlotItem(
            pos=np.array(points),
            color=(0.3, 0.5, 1.0, 0.6),
            size=0.5,
            pxMode=False
        )
        self.view.addItem(self.particles_item)

    def clear_visualization(self):
        if self.population_item:
            self.view.removeItem(self.population_item)
            self.population_item = None
        if self.particles_item:
            self.view.removeItem(self.particles_item)
            self.particles_item = None
        if self.trajectory_line:
            self.view.removeItem(self.trajectory_line)
            self.trajectory_line = None

    def initialize_ga_population(self):
        self.ga_population = []
        for _ in range(self.ga_population_size):
            x = random.uniform(self.xmin, self.xmax)
            y = random.uniform(self.ymin, self.ymax)
            self.ga_population.append((x, y))

    def evaluate_fitness(self, population):
        fitness = []
        for x, y in population:
            f = self.current_func(x, y)
            fitness.append(f)
        return fitness

    def select_parents(self, population, fitness):
        sorted_indices = np.argsort(fitness)
        elite = [population[i] for i in sorted_indices[:self.ga_elite_size]]

        parents = []
        tournament_size = 3
        for _ in range(self.ga_population_size - self.ga_elite_size):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = tournament_indices[np.argmin([fitness[i] for i in tournament_indices])]
            parents.append(population[best_idx])

        return elite, parents

    def crossover(self, parent1, parent2):
        if random.random() < self.ga_crossover_rate:
            alpha = random.random()
            child1_x = alpha * parent1[0] + (1 - alpha) * parent2[0]
            child1_y = alpha * parent1[1] + (1 - alpha) * parent2[1]
            child2_x = (1 - alpha) * parent1[0] + alpha * parent2[0]
            child2_y = (1 - alpha) * parent1[1] + alpha * parent2[1]
            return [(child1_x, child1_y), (child2_x, child2_y)]
        return [parent1, parent2]

    def mutate(self, individual):
        if random.random() < self.ga_mutation_rate:
            mutation_strength = 0.1 * (self.xmax - self.xmin)
            x = individual[0] + random.gauss(0, mutation_strength)
            y = individual[1] + random.gauss(0, mutation_strength)
            x = max(self.xmin, min(self.xmax, x))
            y = max(self.ymin, min(self.ymax, y))
            return (x, y)
        return individual

    def create_next_generation(self, elite, parents):
        next_generation = list(elite)

        for i in range(0, len(parents) - 1, 2):
            if i + 1 < len(parents):
                children = self.crossover(parents[i], parents[i + 1])
                for child in children:
                    child = self.mutate(child)
                    next_generation.append(child)
            else:
                child = self.mutate(parents[i])
                next_generation.append(child)

        while len(next_generation) < self.ga_population_size:
            x = random.uniform(self.xmin, self.xmax)
            y = random.uniform(self.ymin, self.ymax)
            next_generation.append((x, y))

        return next_generation[:self.ga_population_size]

    def run_ga_phase(self):
        self.window.textEdit.append("=" * 60)
        self.window.textEdit.append("ФАЗА 1: ГЕНЕТИЧЕСКИЙ АЛГОРИТМ (глобальный поиск)")
        self.window.textEdit.append("=" * 60)
        self.window.textEdit.append(f"Размер популяции: {self.ga_population_size}")
        self.window.textEdit.append(f"Поколений: {self.ga_max_generations}")
        self.window.textEdit.append(f"Границы: x∈[{self.xmin},{self.xmax}], y∈[{self.ymin},{self.ymax}]")
        self.window.textEdit.append("-" * 60)

        self.initialize_ga_population()
        self.show_ga_population(self.ga_population)
        QApplication.processEvents()

        best_fitness_history = []
        update_interval = max(1, self.ga_max_generations // 20)

        for generation in range(self.ga_max_generations):
            if not self.running:
                return False

            self.current_generation = generation

            fitness = self.evaluate_fitness(self.ga_population)

            best_idx = np.argmin(fitness)
            best_x, best_y = self.ga_population[best_idx]
            best_f = fitness[best_idx]

            if best_f < self.best_value:
                self.best_value = best_f
                self.best_solution = (best_x, best_y, best_f)
                self.show_best_point(best_x, best_y)
                self.add_trajectory_point(best_x, best_y)

            best_fitness_history.append(best_f)

            if generation % update_interval == 0 or generation == self.ga_max_generations - 1:
                self.window.textEdit.append(
                    f"Поколение {generation + 1}/{self.ga_max_generations}: "
                    f"лучшее значение = {best_f:.8f} в точке ({best_x:.6f}, {best_y:.6f})"
                )

            elite, parents = self.select_parents(self.ga_population, fitness)
            self.ga_population = self.create_next_generation(elite, parents)

            self.show_ga_population(self.ga_population)
            self.show_point(best_x, best_y)

            QApplication.processEvents()
            time.sleep(0.02)

        self.window.textEdit.append("-" * 60)
        self.window.textEdit.append(f"Лучшее решение GA: f = {self.best_value:.8f}")
        self.window.textEdit.append(f"Точка: ({self.best_solution[0]:.6f}, {self.best_solution[1]:.6f})")
        self.window.textEdit.append("")

        return True

    class PSOParticle:
        def __init__(self, position, velocity):
            self.current_position = position.copy()
            self.velocity = velocity.copy()
            self.best_position = position.copy()
            self.best_value = np.inf

        def update_best(self, value, position):
            if value < self.best_value:
                self.best_value = value
                self.best_position = position.copy()

    def initialize_pso_particles(self, seed_position=None):
        self.pso_particles_list = []
        velocity_range_x = (self.xmax - self.xmin) * 0.2
        velocity_range_y = (self.ymax - self.ymin) * 0.2

        # Используем лучшую точку GA как начальную для одной частицы
        if seed_position is None and self.best_solution:
            seed_position = (self.best_solution[0], self.best_solution[1])

        for i in range(self.pso_particles):
            if i == 0 and seed_position:
                x, y = seed_position
            else:
                # Остальные частицы вокруг лучшей точки
                if seed_position:
                    spread = 0.1 * (self.xmax - self.xmin)
                    x = seed_position[0] + random.uniform(-spread, spread)
                    y = seed_position[1] + random.uniform(-spread, spread)
                    x = max(self.xmin, min(self.xmax, x))
                    y = max(self.ymin, min(self.ymax, y))
                else:
                    x = random.uniform(self.xmin, self.xmax)
                    y = random.uniform(self.ymin, self.ymax)

            position = np.array([x, y])

            vx = random.uniform(-velocity_range_x, velocity_range_x)
            vy = random.uniform(-velocity_range_y, velocity_range_y)
            velocity = np.array([vx, vy])

            particle = self.PSOParticle(position, velocity)
            value = self.objective(x, y)

            particle.update_best(value, position)

            if value < self.pso_global_best_value:
                self.pso_global_best_value = value
                self.pso_global_best = position.copy()

            self.pso_particles_list.append(particle)

    def pso_update_velocity(self, particle):
        velo_ratio = self.pso_local_ratio + self.pso_global_ratio
        denominator = abs(2.0 - velo_ratio - np.sqrt(velo_ratio ** 2 - 4.0 * velo_ratio))
        if denominator < 1e-10:
            common_ratio = self.pso_velocity_ratio
        else:
            common_ratio = 2.0 * self.pso_velocity_ratio / denominator

        r1 = np.random.rand(2)
        r2 = np.random.rand(2)

        cognitive = self.pso_local_ratio * r1 * (particle.best_position - particle.current_position)
        social = self.pso_global_ratio * r2 * (self.pso_global_best - particle.current_position)

        new_velocity = common_ratio * (particle.velocity + cognitive + social)

        return new_velocity

    def run_pso_phase(self):
        self.window.textEdit.append("=" * 60)
        self.window.textEdit.append("ФАЗА 2: МЕТОД РОЯ ЧАСТИЦ (локальная оптимизация)")
        self.window.textEdit.append("=" * 60)
        self.window.textEdit.append(f"Количество частиц: {self.pso_particles}")
        self.window.textEdit.append(f"Итераций: {self.pso_max_iterations}")
        self.window.textEdit.append(f"ω = {self.pso_velocity_ratio}, φ₁ = {self.pso_local_ratio}, φ₂ = {self.pso_global_ratio}")
        self.window.textEdit.append(f"Начальная точка (из GA): ({self.best_solution[0]:.6f}, {self.best_solution[1]:.6f})")
        self.window.textEdit.append("-" * 60)

        self.pso_global_best_value = np.inf
        self.pso_global_best = None

        self.clear_visualization()
        self.initialize_pso_particles((self.best_solution[0], self.best_solution[1]))
        self.show_pso_particles(self.pso_particles_list)

        if self.best_point_item:
            self.view.removeItem(self.best_point_item)
            self.best_point_item = None

        self.show_best_point(self.best_solution[0], self.best_solution[1])

        QApplication.processEvents()

        best_value_before_pso = self.best_value

        for iteration in range(self.pso_max_iterations):
            if not self.running:
                return False

            for particle in self.pso_particles_list:
                particle.velocity = self.pso_update_velocity(particle)

                new_position = particle.current_position + particle.velocity
                new_position = np.clip(new_position, [self.xmin, self.ymin], [self.xmax, self.ymax])

                particle.current_position = new_position

                value = self.objective(new_position[0], new_position[1])

                particle.update_best(value, new_position)

                if value < self.pso_global_best_value:
                    self.pso_global_best_value = value
                    self.pso_global_best = new_position.copy()

                    if value < self.best_value:
                        self.best_value = value
                        self.best_solution = (new_position[0], new_position[1], value)
                        self.add_trajectory_point(new_position[0], new_position[1])
                        self.show_best_point(new_position[0], new_position[1])

            if iteration % max(1, self.pso_max_iterations // 10) == 0 or iteration == self.pso_max_iterations - 1:
                self.window.textEdit.append(
                    f"Итерация PSO {iteration + 1}/{self.pso_max_iterations}: "
                    f"лучшее значение = {self.pso_global_best_value:.8f}"
                )

            self.show_pso_particles(self.pso_particles_list)
            if self.pso_global_best is not None:
                self.show_point(self.pso_global_best[0], self.pso_global_best[1])

            QApplication.processEvents()
            time.sleep(0.02)

        improvement = best_value_before_pso - self.best_value
        self.window.textEdit.append("-" * 60)
        self.window.textEdit.append(f"Улучшение после PSO: {improvement:.2e}")

        return True

    def run(self):
        if self.current_func is None:
            self.window.textEdit.append("Сначала постройте поверхность")
            return

        self.running = True
        self.best_value = np.inf
        self.best_solution = None
        self.trajectory_points = []

        self.window.textEdit.append("")
        self.window.textEdit.append("╔" + "=" * 58 + "╗")
        self.window.textEdit.append("║     ГИБРИДНЫЙ АЛГОРИТМ GA + PSO     ║")
        self.window.textEdit.append("╚" + "=" * 58 + "╝")
        self.window.textEdit.append("")
        self.window.textEdit.append("Тип гибридизации: последовательный (препроцессор/постпроцессор)")
        self.window.textEdit.append("GA → глобальный поиск, PSO → локальная оптимизация")
        self.window.textEdit.append("")

        success = self.run_ga_phase()

        if success and self.running and self.best_solution:
            self.run_pso_phase()

        self.window.textEdit.append("")
        self.window.textEdit.append("=" * 60)
        self.window.textEdit.append("РЕЗУЛЬТАТ ГИБРИДНОЙ ОПТИМИЗАЦИИ")
        self.window.textEdit.append("=" * 60)
        if self.best_solution:
            self.window.textEdit.append(f"x* = {self.best_solution[0]:.8f}")
            self.window.textEdit.append(f"y* = {self.best_solution[1]:.8f}")
            self.window.textEdit.append(f"f(x*,y*) = {self.best_solution[2]:.8f}")

        self.show_best_point(self.best_solution[0], self.best_solution[1])
        self.show_point(self.best_solution[0], self.best_solution[1])

        return self.best_solution

    def stop(self):
        self.running = False
        self.window.textEdit.append("Гибридный алгоритм остановлен")

    def reset(self):
        self.running = False
        self.current_phase = "GA"
        self.current_generation = 0
        self.best_solution = None
        self.best_value = np.inf
        self.ga_population = []
        self.pso_particles_list = []
        self.pso_global_best = None
        self.trajectory_points = []

        self.clear_visualization()

        if self.best_point_item:
            self.view.removeItem(self.best_point_item)
            self.best_point_item = None
        if self.temp_point_item:
            self.view.removeItem(self.temp_point_item)
            self.temp_point_item = None
        if self.trajectory_line:
            self.view.removeItem(self.trajectory_line)
            self.trajectory_line = None

        self.window.textEdit.append("Гибридный алгоритм сброшен")