# src/core/optimization_visualizer.py
import numpy as np
import pyqtgraph.opengl as gl


class OptimizationVisualizer:

    def __init__(self, view_widget):
        self.view = view_widget
        self.points = []
        self.arrows = []
        self.scatter_item = None
        self.line_item = None
        self.current_point_item = None
        self.arrow_items = []
        self.trajectory_color = (1.0, 0.0, 0.0, 1.0)
        self.current_point_color = (0.0, 1.0, 0.0, 1.0)
        self.arrow_color = (1.0, 0.5, 0.0, 0.8)

    def clear(self):
        if self.scatter_item:
            self.view.removeItem(self.scatter_item)
            self.scatter_item = None

        if self.line_item:
            self.view.removeItem(self.line_item)
            self.line_item = None

        if self.current_point_item:
            self.view.removeItem(self.current_point_item)
            self.current_point_item = None

        for arrow in self.arrow_items:
            self.view.removeItem(arrow)
        self.arrow_items = []

        self.points = []
        self.arrows = []

    def add_point(self, point, is_current=False):
        self.points.append(point)
        self._update_trajectory()

        if is_current:
            self._update_current_point(point)

    def add_arrow(self, start_point, end_point):
        self.arrows.append((start_point, end_point))
        self._update_arrows()

    def _update_trajectory(self):
        if len(self.points) < 2:
            return

        points = np.array(self.points)

        if self.line_item:
            self.view.removeItem(self.line_item)

        self.line_item = gl.GLLinePlotItem(
            pos=points,
            color=self.trajectory_color,
            width=2,
            antialias=True
        )
        self.view.addItem(self.line_item)

        if self.scatter_item:
            self.view.removeItem(self.scatter_item)

        n_points = len(points)
        colors = np.zeros((n_points, 4))
        colors[:, :3] = self.trajectory_color[:3]
        colors[:, 3] = 0.5

        self.scatter_item = gl.GLScatterPlotItem(
            pos=points,
            color=colors,
            size=1,
            pxMode=False
        )
        self.view.addItem(self.scatter_item)

    def _update_current_point(self, point):
        if self.current_point_item:
            self.view.removeItem(self.current_point_item)

        self.current_point_item = gl.GLScatterPlotItem(
            pos=np.array([point]),
            color=self.current_point_color,
            size=2,
            pxMode=False
        )
        self.view.addItem(self.current_point_item)

    def _update_arrows(self):
        for arrow in self.arrow_items:
            self.view.removeItem(arrow)
        self.arrow_items = []

        for start, end in self.arrows[-5:]:
            # Линия стрелки
            line_item = gl.GLLinePlotItem(
                pos=np.array([start, end]),
                color=self.arrow_color,
                width=1.5,
                antialias=True
            )
            self.view.addItem(line_item)
            self.arrow_items.append(line_item)