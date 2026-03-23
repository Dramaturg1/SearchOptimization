# src/utils/point_generator.py
import numpy as np
import random


class PointGenerator:

    def __init__(self, xmin=-5, xmax=5, ymin=-5, ymax=5):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def update_bounds(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def generate_single(self):
        x = random.uniform(self.xmin, self.xmax)
        y = random.uniform(self.ymin, self.ymax)
        return x, y

    def generate_multiple(self, count=100):
        points = []
        for _ in range(count):
            x = random.uniform(self.xmin, self.xmax)
            y = random.uniform(self.ymin, self.ymax)
            points.append((x, y))
        return points

    def generate_grid(self, nx=5, ny=5):
        x = np.linspace(self.xmin, self.xmax, nx)
        y = np.linspace(self.ymin, self.ymax, ny)
        points = []
        for xi in x:
            for yi in y:
                points.append((xi, yi))
        return points

    def generate_with_center_bias(self, count=1, center_x=0, center_y=0, spread=0.3):
        points = []
        for _ in range(count):
            x = center_x + random.uniform(-spread, spread) * (self.xmax - self.xmin)
            y = center_y + random.uniform(-spread, spread) * (self.ymax - self.ymin)
            x = max(self.xmin, min(self.xmax, x))
            y = max(self.ymin, min(self.ymax, y))
            points.append((x, y))
        return points