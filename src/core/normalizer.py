# src/core/normalizer.py
import numpy as np


class PointNormalizer:
    """
    Класс для нормирования точек между координатным пространством функции
    и пространством отображения
    """

    def __init__(self, xmin, xmax, ymin, ymax, z_min, z_max):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.z_min = z_min
        self.z_max = z_max

        # Масштабирование Z как в plotter.py
        z_range = z_max - z_min
        if z_range == 0:
            self.z_scale = 1.0
        else:
            self.z_scale = 10 / z_range

        self.z_center = (z_max + z_min) / 2

    def to_display_coords(self, x, y, z):
        """Конвертирует координаты в пространство отображения"""
        disp_x = float(x)
        disp_y = float(y)
        disp_z = (float(z) - self.z_center) * self.z_scale
        return np.array([disp_x, disp_y, disp_z])

    def to_function_coords(self, disp_x, disp_y, disp_z):
        """Конвертирует координаты из пространства отображения"""
        func_x = float(disp_x)
        func_y = float(disp_y)
        func_z = (float(disp_z) / self.z_scale) + self.z_center
        return np.array([func_x, func_y, func_z])

    def get_z_offset(self):
        return -self.z_center

    def get_z_scale(self):
        return self.z_scale