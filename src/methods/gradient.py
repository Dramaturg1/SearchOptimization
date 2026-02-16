# src/methods/gradient.py
import numpy as np
from scipy.optimize import minimize_scalar


class SteepestDescent:
    """
    Метод наискорейшего спуска для поиска минимума функции.
    """

    def __init__(self, func, grad, x0, eps=1e-6, M=100, delta=1e-7):
        self.func = func  # функция f(x,y)
        self.grad = grad  # градиент
        self.x = np.array(x0, dtype=float)
        self.eps = eps
        self.M = M
        self.delta = delta

        self.k = 0
        self.prev_x = None
        self.prev_f = None
        self.history = [self.x.copy()]
        self.double_check_count = 0

    def step(self):
        """Выполняет одну итерацию метода"""
        grad = self.grad(self.x)
        grad_norm = np.linalg.norm(grad)

        # Проверка по норме градиента
        if grad_norm < self.eps:
            return self.x.copy(), True, f"Норма градиента меньше eps = {self.eps}"

        # Проверка по числу итераций
        if self.k >= self.M:
            return self.x.copy(), True, f"Достигнуто максимальное число итераций M = {self.M}"

        # Поиск оптимального шага
        def func_for_t(t):
            x_test = self.x - t * grad
            return self.func(x_test[0], x_test[1])

        result = minimize_scalar(func_for_t, bounds=(0, 1), method='bounded')
        t = result.x

        # Вычисляем новую точку
        x_new = self.x - t * grad
        f_new = self.func(x_new[0], x_new[1])
        f_old = self.func(self.x[0], self.x[1])

        # Сохраняем предыдущее состояние
        self.prev_x = self.x.copy()
        self.prev_f = f_old

        # Обновляем текущую точку
        self.x = x_new
        self.k += 1
        self.history.append(self.x.copy())

        # Проверка двойного условия остановки
        if self.prev_x is not None and self.prev_f is not None:
            x_diff = np.linalg.norm(self.x - self.prev_x)
            f_diff = abs(f_new - self.prev_f)

            if x_diff < self.delta and f_diff < self.delta:
                self.double_check_count += 1
                if self.double_check_count >= 2:
                    return self.x.copy(), True, "Двойное условие остановки выполнено дважды подряд"
            else:
                self.double_check_count = 0

        return self.x.copy(), False, "Итерация выполнена успешно"

    def reset(self, x0=None):
        """Сброс метода"""
        if x0 is not None:
            self.x = np.array(x0, dtype=float)
        self.k = 0
        self.prev_x = None
        self.prev_f = None
        self.double_check_count = 0
        self.history = [self.x.copy()]