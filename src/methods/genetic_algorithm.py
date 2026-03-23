# src/methods/genetic_algorithm.py
import numpy as np
from PySide6.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import time
import random


class GeneticAlgorithm:
    def __init__(self, view, current_func, current_zmin, current_zmax, point_item, window):
        self.view = view
        self.current_func = current_func
        self.current_zmin = current_zmin
        self.current_zmax = current_zmax
        self.point_item = point_item
        self.window = window

        self.running = False
        self.population_item = None
        self.best_point_item = None

        self.population_size = 50
        self.max_iterations = 100
        self.use_mutation = True
        self.use_crossover = True
        self.use_convergence = True

        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 2

        self.xmin = -5
        self.xmax = 5
        self.ymin = -5
        self.ymax = 5

        self.best_solution = None
        self.best_fitness = float('inf')

        self.points_cache = None
        self.colors_cache = None

    def set_function(self, func, zmin, zmax):
        self.current_func = func
        self.current_zmin = zmin
        self.current_zmax = zmax

    def update_bounds(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def set_parameters(self, population_size, max_iterations, use_mutation, use_crossover, use_convergence):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.use_mutation = use_mutation
        self.use_crossover = use_crossover
        self.use_convergence = use_convergence

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

    def show_population(self, population):
        if len(population) == 0:
            return

        points = []
        colors = []

        f_values = []
        for x, y in population:
            z = self.current_func(x, y)
            f_values.append(z)

        f_min = min(f_values)
        f_max = max(f_values)

        for (x, y), z in zip(population, f_values):
            z_vis = self.z_to_vis(z)
            points.append([x, y, z_vis])

            if f_max > f_min:
                norm = (z - f_min) / (f_max - f_min)
            else:
                norm = 0
            colors.append([norm, 1 - norm, 0, 0.8])

        points = np.array(points)
        colors = np.array(colors)

        if self.population_item is not None:
            self.population_item.setData(pos=points, color=colors)
        else:
            self.population_item = gl.GLScatterPlotItem(
                pos=points,
                color=colors,
                size=0.5,
                pxMode=False
            )
            self.view.addItem(self.population_item)

    def show_best_point(self, x, y):
        if self.current_func is None:
            return

        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])

        if self.best_point_item is not None:
            self.best_point_item.setData(pos=pos)
        else:
            self.best_point_item = gl.GLScatterPlotItem(
                pos=pos,
                color=(0, 1, 0, 1),
                size=1.0,
                pxMode=False
            )
            self.view.addItem(self.best_point_item)

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            x = random.uniform(self.xmin, self.xmax)
            y = random.uniform(self.ymin, self.ymax)
            population.append((x, y))
        return population

    def evaluate_fitness(self, population):
        fitness = []
        for x, y in population:
            f = self.current_func(x, y)
            fitness.append(f)
        return fitness

    def select_parents(self, population, fitness):
        sorted_indices = np.argsort(fitness)
        elite = [population[i] for i in sorted_indices[:self.elite_size]]

        parents = []
        tournament_size = 3
        for _ in range(self.population_size - self.elite_size):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = tournament_indices[np.argmin([fitness[i] for i in tournament_indices])]
            parents.append(population[best_idx])

        return elite, parents

    def crossover(self, parent1, parent2):
        if self.use_crossover and random.random() < self.crossover_rate:
            alpha = random.random()
            child1_x = alpha * parent1[0] + (1 - alpha) * parent2[0]
            child1_y = alpha * parent1[1] + (1 - alpha) * parent2[1]
            child2_x = (1 - alpha) * parent1[0] + alpha * parent2[0]
            child2_y = (1 - alpha) * parent1[1] + alpha * parent2[1]
            return [(child1_x, child1_y), (child2_x, child2_y)]
        else:
            return [parent1, parent2]

    def mutate(self, individual):
        if self.use_mutation and random.random() < self.mutation_rate:
            mutation_strength = 0.1 * (self.xmax - self.xmin)
            x = individual[0] + random.gauss(0, mutation_strength)
            y = individual[1] + random.gauss(0, mutation_strength)
            x = max(self.xmin, min(self.xmax, x))
            y = max(self.ymin, min(self.ymax, y))
            return (x, y)
        return individual

    def convergence_operator(self, population, fitness):
        if not self.use_convergence:
            return population

        best_idx = np.argmin(fitness)
        best_x, best_y = population[best_idx]

        new_population = []
        for i, (x, y) in enumerate(population):
            if i != best_idx:
                alpha = 0.5
                x_new = x + alpha * (best_x - x)
                y_new = y + alpha * (best_y - y)
                x_new = max(self.xmin, min(self.xmax, x_new))
                y_new = max(self.ymin, min(self.ymax, y_new))
                new_population.append((x_new, y_new))
            else:
                new_population.append((x, y))

        return new_population

    def create_next_generation(self, elite, parents):
        next_generation = list(elite)

        for i in range(0, len(parents) - 1, 2):
            if i + 1 < len(parents):
                children = self.crossover(parents[i], parents[i + 1])
                for child in children:
                    child = self.mutate(child)
                    next_generation.append(child)
            else:
                child = parents[i]
                child = self.mutate(child)
                next_generation.append(child)

        while len(next_generation) < self.population_size:
            x = random.uniform(self.xmin, self.xmax)
            y = random.uniform(self.ymin, self.ymax)
            next_generation.append((x, y))

        next_generation = next_generation[:self.population_size]

        return next_generation

    def run(self):
        if self.current_func is None:
            self.window.textEdit.append("Сначала постройте поверхность")
            return

        self.window.textEdit.append("ГЕНЕТИЧЕСКИЙ АЛГОРИТМ")
        self.window.textEdit.append(f"Размер популяции: {self.population_size}")
        self.window.textEdit.append(f"Максимум итераций: {self.max_iterations}")
        self.window.textEdit.append(f"Используемые операторы:")
        if self.use_mutation:
            self.window.textEdit.append(f"  - Мутация (вероятность: {self.mutation_rate})")
        if self.use_crossover:
            self.window.textEdit.append(f"  - Скрещивание (вероятность: {self.crossover_rate})")
        if self.use_convergence:
            self.window.textEdit.append(f"  - Сближение")
        self.window.textEdit.append(f"Область поиска: x ∈ [{self.xmin}, {self.xmax}], y ∈ [{self.ymin}, {self.ymax}]")
        self.window.textEdit.append("")

        self.running = True

        population = self.initialize_population()
        self.show_population(population)
        QApplication.processEvents()

        best_fitness_history = []
        update_interval = max(1, self.max_iterations // 50)

        for iteration in range(self.max_iterations):
            if not self.running:
                self.window.textEdit.append("\nАлгоритм остановлен пользователем")
                break

            fitness = self.evaluate_fitness(population)

            best_idx = np.argmin(fitness)
            best_x, best_y = population[best_idx]
            best_f = fitness[best_idx]

            if best_f < self.best_fitness:
                self.best_fitness = best_f
                self.best_solution = (best_x, best_y, best_f)
                self.show_best_point(best_x, best_y)

            best_fitness_history.append(best_f)

            if iteration % update_interval == 0 or iteration == self.max_iterations - 1:
                self.window.textEdit.append(f"Итерация {iteration + 1}/{self.max_iterations}:")
                self.window.textEdit.append(f"  Лучшая точка: ({best_x:.6f}, {best_y:.6f})")
                self.window.textEdit.append(f"  Значение функции: {best_f:.6f}")

            elite, parents = self.select_parents(population, fitness)
            next_generation = self.create_next_generation(elite, parents)

            if self.use_convergence:
                fitness_next = self.evaluate_fitness(next_generation)
                next_generation = self.convergence_operator(next_generation, fitness_next)

            population = next_generation

            self.show_population(population)
            self.show_point(best_x, best_y)

            QApplication.processEvents()
            time.sleep(0.033)

        self.window.textEdit.append("РЕЗУЛЬТАТЫ РАБОТЫ АЛГОРИТМА")

        if self.best_solution:
            x, y, f = self.best_solution
            self.window.textEdit.append(f"Найденный минимум:")
            self.window.textEdit.append(f"  x = {x:.8f}")
            self.window.textEdit.append(f"  y = {y:.8f}")
            self.window.textEdit.append(f"  f = {f:.8f}")

            self.show_best_point(x, y)
            self.show_point(x, y)

    def stop(self):
        self.running = False
        self.window.textEdit.append("Остановка генетического алгоритма...")

    def reset(self):
        self.running = False

        if self.population_item:
            self.view.removeItem(self.population_item)
            self.population_item = None

        if self.best_point_item:
            self.view.removeItem(self.best_point_item)
            self.best_point_item = None

        self.best_solution = None
        self.best_fitness = float('inf')

        self.window.textEdit.clear()