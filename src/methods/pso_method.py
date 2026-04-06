# src/methods/pso_method.py
import numpy as np
from PySide6.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import time


class Particle:
    """Класс, описывающий одну частицу"""

    def __init__(self, position, velocity):
        self.current_position = position.copy()
        self.velocity = velocity.copy()
        self.best_position = position.copy()
        self.best_value = np.inf

    def update_best(self, value, position):
        if value < self.best_value:
            self.best_value = value
            self.best_position = position.copy()


class PSOMethod:
    def __init__(self, view, current_func, current_zmin, current_zmax, window):
        self.view = view
        self.current_func = current_func
        self.current_zmin = current_zmin
        self.current_zmax = current_zmax
        self.window = window

        self.running = False
        self.step_mode = False
        self.particles = []
        self.global_best_position = None
        self.global_best_value = np.inf
        self.current_generation = 0
        self.max_generations = 0
        self.dimension = 2

        self.particles_item = None
        self.best_particle_item = None
        self.temp_point_item = None

        # Параметры алгоритма (как в методичке)
        self.current_velocity_ratio = 0.5  # ω
        self.local_velocity_ratio = 2.0  # φ₁
        self.global_velocity_ratio = 5.0  # φ₂
        self.common_ratio = None  # χ

        # Операторы (для совместимости с UI)
        self.use_mutation = False
        self.use_crossover = False
        self.use_convergence = False

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

    def show_point(self, x, y):
        if self.current_func is None:
            return
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])
        if self.temp_point_item:
            self.view.removeItem(self.temp_point_item)
        self.temp_point_item = gl.GLScatterPlotItem(pos=pos, size=8, color=(1, 0, 0, 1))
        self.view.addItem(self.temp_point_item)

    def update_particles_visualization(self):
        if self.particles_item:
            self.view.removeItem(self.particles_item)

        if len(self.particles) == 0:
            return

        positions = []
        colors = []
        for p in self.particles:
            x, y = p.current_position
            z = self.current_func(x, y)
            z_vis = self.z_to_vis(z)
            positions.append([x, y, z_vis])
            colors.append((0.5, 0.5, 1.0, 0.7))

        self.particles_item = gl.GLScatterPlotItem(
            pos=np.array(positions),
            color=np.array(colors),
            size=1,
            pxMode=False
        )
        self.view.addItem(self.particles_item)

    def update_best_visualization(self):
        if self.best_particle_item:
            self.view.removeItem(self.best_particle_item)

        if self.global_best_position is None:
            return

        x, y = self.global_best_position
        z = self.current_func(x, y)
        z_vis = self.z_to_vis(z)
        pos = np.array([[x, y, z_vis]])

        self.best_particle_item = gl.GLScatterPlotItem(
            pos=pos,
            size=3,
            color=(1, 0.8, 0, 1),
            pxMode=False
        )
        self.view.addItem(self.best_particle_item)

    def calculate_common_ratio(self):
        """
        Расчёт commonRatio (χ) по формуле из методички:
        veloRatio = φ₁ + φ₂
        commonRatio = 2 * ω / |2 - veloRatio - √(veloRatio² - 4*veloRatio)|
        """
        velo_ratio = self.local_velocity_ratio + self.global_velocity_ratio
        denominator = abs(2.0 - velo_ratio - np.sqrt(velo_ratio ** 2 - 4.0 * velo_ratio))
        if denominator < 1e-10:
            return self.current_velocity_ratio
        common_ratio = 2.0 * self.current_velocity_ratio / denominator
        return common_ratio

    def _get_penalty(self, position, ratio=10000.0):
        """
        Штрафная функция из методички (аналог _getPenalty)
        penalty = sum(ratio * |coord - bound|) для координат вне границ
        """
        penalty = 0.0
        # Штраф за выход за minvalues
        for coord, minval in zip(position, [self.xmin, self.ymin]):
            if coord < minval:
                penalty += ratio * abs(coord - minval)
        # Штраф за выход за maxvalues
        for coord, maxval in zip(position, [self.xmax, self.ymax]):
            if coord > maxval:
                penalty += ratio * abs(coord - maxval)
        return penalty

    def initialize_particles(self, n_particles):
        """Инициализация роя частиц со случайными позициями и скоростями"""
        self.particles = []
        self.global_best_position = None
        self.global_best_value = np.inf

        # Диапазон скоростей: от -range до range
        velocity_range_x = (self.xmax - self.xmin) * 0.5
        velocity_range_y = (self.ymax - self.ymin) * 0.5

        for _ in range(n_particles):
            # Случайная позиция в области поиска
            x = np.random.uniform(self.xmin, self.xmax)
            y = np.random.uniform(self.ymin, self.ymax)
            position = np.array([x, y])

            # Случайная начальная скорость
            vx = np.random.uniform(-velocity_range_x, velocity_range_x)
            vy = np.random.uniform(-velocity_range_y, velocity_range_y)
            velocity = np.array([vx, vy])

            particle = Particle(position, velocity)

            # Вычисление значения со штрафом (как в методичке)
            value = self.current_func(x, y)
            penalty = self._get_penalty(position, 10000.0)
            total_value = value + penalty

            particle.update_best(total_value, position)

            if total_value < self.global_best_value:
                self.global_best_value = total_value
                self.global_best_position = position.copy()

            self.particles.append(particle)

        self.window.textEdit.append(f"Рой инициализирован: {n_particles} частиц")
        self.window.textEdit.append(f"Границы: x∈[{self.xmin},{self.xmax}], y∈[{self.ymin},{self.ymax}]")

    def update_velocity(self, particle):
        """
        Обновление скорости частицы по ТОЧНОЙ формуле из методички:

        commonRatio = 2 * ω / |2 - φ - √(φ² - 4φ)|
        v_new = commonRatio * (v + φ₁*r₁*(p - x) + φ₂*r₂*(g - x))

        где:
        - φ = φ₁ + φ₂
        - r₁, r₂ — случайные числа в интервале (0, 1)
        """
        # Случайные векторы r₁ и r₂ в интервале (0, 1)
        r1 = np.random.rand(self.dimension)
        r2 = np.random.rand(self.dimension)

        # Компонента когнитивной части (ностальгия) - φ₁*r₁*(p - x)
        cognitive = self.local_velocity_ratio * r1 * (particle.best_position - particle.current_position)

        # Компонента социальной части - φ₂*r₂*(g - x)
        social = self.global_velocity_ratio * r2 * (self.global_best_position - particle.current_position)

        # Точная формула из методички
        new_velocity = self.common_ratio * (particle.velocity + cognitive + social)

        return new_velocity

    def mutation(self, position, mutation_rate=0.1):
        """Мутация: случайное возмущение позиции"""
        x, y = position
        if np.random.rand() < mutation_rate:
            x += np.random.uniform(-(self.xmax - self.xmin) * 0.1, (self.xmax - self.xmin) * 0.1)
            y += np.random.uniform(-(self.ymax - self.ymin) * 0.1, (self.ymax - self.ymin) * 0.1)
            x = np.clip(x, self.xmin, self.xmax)
            y = np.clip(y, self.ymin, self.ymax)
        return np.array([x, y])

    def crossover(self, p1, p2, crossover_rate=0.8):
        """Скрещивание: взвешенное среднее двух позиций"""
        if np.random.rand() < crossover_rate:
            alpha = np.random.rand()
            new1 = alpha * p1 + (1 - alpha) * p2
            new2 = alpha * p2 + (1 - alpha) * p1
            new1 = np.clip(new1, [self.xmin, self.ymin], [self.xmax, self.ymax])
            new2 = np.clip(new2, [self.xmin, self.ymin], [self.xmax, self.ymax])
            return new1, new2
        return p1.copy(), p2.copy()

    def convergence_operator(self, position, conv_rate=0.3):
        """Сближение: смещение позиции к глобальному лучшему решению"""
        if self.global_best_position is None:
            return position
        if np.random.rand() < conv_rate:
            beta = np.random.rand()
            new_pos = position + beta * (self.global_best_position - position)
            new_pos = np.clip(new_pos, [self.xmin, self.ymin], [self.xmax, self.ymax])
            return new_pos
        return position

    def run(self, n_particles=30, max_iter=100):
        """
        Запуск PSO с параметрами из методички:
        - currentVelocityRatio = 0.5 (ω)
        - localVelocityRatio = 2.0 (φ₁)
        - globalVelocityRatio = 5.0 (φ₂)
        """
        # Расчёт commonRatio по формуле из методички
        self.common_ratio = self.calculate_common_ratio()

        self.initialize_particles(n_particles)
        self.current_generation = 0
        self.max_generations = max_iter
        self.running = True

        phi = self.local_velocity_ratio + self.global_velocity_ratio

        self.window.textEdit.append("ЗАПУСК АЛГОРИТМА РОЯ ЧАСТИЦ (PSO)")
        self.window.textEdit.append(f"Параметры (из методички):")
        self.window.textEdit.append(f"  ω (currentVelocityRatio) = {self.current_velocity_ratio}")
        self.window.textEdit.append(f"  φ₁ (localVelocityRatio)  = {self.local_velocity_ratio}")
        self.window.textEdit.append(f"  φ₂ (globalVelocityRatio) = {self.global_velocity_ratio}")
        self.window.textEdit.append(f"  φ = φ₁+φ₂ = {phi}")
        self.window.textEdit.append(f"  χ (commonRatio) = {self.common_ratio:.6f}")
        self.window.textEdit.append(f"Частиц: {n_particles}, поколений: {max_iter}")
        self.window.textEdit.append(f"Операторы: мутация={self.use_mutation}, "
                                    f"скрещивание={self.use_crossover}, сближение={self.use_convergence}")

        self.update_particles_visualization()
        self.update_best_visualization()
        QApplication.processEvents()

        for generation in range(max_iter):
            if not self.running:
                break

            self.current_generation = generation

            # Обновление скоростей и позиций всех частиц
            for i, particle in enumerate(self.particles):
                # Обновление скорости по формуле из методички
                particle.velocity = self.update_velocity(particle)

                # Обновление позиции
                new_position = particle.current_position + particle.velocity

                # Применение операторов (если включены)
                if self.use_mutation:
                    new_position = self.mutation(new_position)
                if self.use_crossover and i % 2 == 0 and i + 1 < len(self.particles):
                    new_position, self.particles[i + 1].current_position = self.crossover(
                        new_position, self.particles[i + 1].current_position
                    )
                if self.use_convergence:
                    new_position = self.convergence_operator(new_position)

                # Обрезаем до границ (альтернатива штрафу, но оставляем и штраф)
                new_position = np.clip(new_position, [self.xmin, self.ymin], [self.xmax, self.ymax])
                particle.current_position = new_position

                # Вычисление значения целевой функции со штрафом (как в методичке)
                value = self.current_func(new_position[0], new_position[1])
                penalty = self._get_penalty(new_position, 10000.0)
                total_value = value + penalty

                # Обновление личного лучшего решения
                particle.update_best(total_value, new_position)

                # Обновление глобального лучшего решения
                if total_value < self.global_best_value:
                    self.global_best_value = total_value
                    self.global_best_position = new_position.copy()

            self.window.textEdit.append(f"Поколение {generation + 1}/{max_iter}: "
                                        f"лучшее значение = {self.global_best_value:.8f}")

            self.update_particles_visualization()
            self.update_best_visualization()
            self.show_point(self.global_best_position[0], self.global_best_position[1])

            QApplication.processEvents()
            time.sleep(0.03)

        # Вывод результата без штрафа (реальное значение функции)
        real_value = self.current_func(self.global_best_position[0], self.global_best_position[1])

        self.window.textEdit.append("РЕЗУЛЬТАТ ОПТИМИЗАЦИИ")
        self.window.textEdit.append(f"x* = {self.global_best_position[0]:.8f}")
        self.window.textEdit.append(f"y* = {self.global_best_position[1]:.8f}")
        self.window.textEdit.append(f"f(x*,y*) = {real_value:.8f}")

        return self.global_best_position[0], self.global_best_position[1], real_value

    def run_step_mode(self, n_particles=30, max_iter=100):
        """Пошаговый режим"""
        self.common_ratio = self.calculate_common_ratio()
        self.initialize_particles(n_particles)
        self.current_generation = 0
        self.max_generations = max_iter
        self.step_mode = True
        self.running = True

        self.update_particles_visualization()
        self.update_best_visualization()
        self.show_point(self.global_best_position[0], self.global_best_position[1])

        phi = self.local_velocity_ratio + self.global_velocity_ratio

        self.window.textEdit.append(f"\nПошаговый режим PSO")
        self.window.textEdit.append(
            f"Параметры: ω={self.current_velocity_ratio}, φ₁={self.local_velocity_ratio}, φ₂={self.global_velocity_ratio}")
        self.window.textEdit.append(f"φ={phi}, χ={self.common_ratio:.6f}")
        self.window.textEdit.append(f"Частиц: {n_particles}, поколений: {max_iter}")
        self.window.textEdit.append(f"Начальное лучшее значение: {self.global_best_value:.8f}")

    def step(self):
        """Один шаг пошагового режима"""
        if not self.step_mode or not self.running:
            return

        if self.current_generation >= self.max_generations:
            self.window.textEdit.append("Достигнуто максимальное число поколений")
            return

        # Обновление скоростей и позиций всех частиц
        for i, particle in enumerate(self.particles):
            particle.velocity = self.update_velocity(particle)

            new_position = particle.current_position + particle.velocity

            if self.use_mutation:
                new_position = self.mutation(new_position)
            if self.use_crossover and i % 2 == 0 and i + 1 < len(self.particles):
                new_position, self.particles[i + 1].current_position = self.crossover(
                    new_position, self.particles[i + 1].current_position
                )
            if self.use_convergence:
                new_position = self.convergence_operator(new_position)

            new_position = np.clip(new_position, [self.xmin, self.ymin], [self.xmax, self.ymax])
            particle.current_position = new_position

            value = self.current_func(new_position[0], new_position[1])
            penalty = self._get_penalty(new_position, 10000.0)
            total_value = value + penalty

            particle.update_best(total_value, new_position)

            if total_value < self.global_best_value:
                self.global_best_value = total_value
                self.global_best_position = new_position.copy()

        self.current_generation += 1

        self.update_particles_visualization()
        self.update_best_visualization()
        self.show_point(self.global_best_position[0], self.global_best_position[1])

        real_value = self.current_func(self.global_best_position[0], self.global_best_position[1])
        self.window.textEdit.append(f"Поколение {self.current_generation}/{self.max_generations}: "
                                    f"лучшее f = {real_value:.8f}")

        if self.current_generation == self.max_generations:
            self.window.textEdit.append(f"\nОптимум: x={self.global_best_position[0]:.8f}, "
                                        f"y={self.global_best_position[1]:.8f}, f={real_value:.8f}")

    def stop(self):
        self.running = False
        self.step_mode = False

    def reset(self):
        self.running = False
        self.step_mode = False
        self.current_generation = 0
        self.particles = []
        self.global_best_position = None
        self.global_best_value = np.inf

        if self.particles_item:
            self.view.removeItem(self.particles_item)
            self.particles_item = None
        if self.best_particle_item:
            self.view.removeItem(self.best_particle_item)
            self.best_particle_item = None
        if self.temp_point_item:
            self.view.removeItem(self.temp_point_item)
            self.temp_point_item = None

        self.window.textEdit.append("Визуализация PSO сброшена")