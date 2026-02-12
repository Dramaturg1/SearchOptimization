import numpy as np
from matplotlib import pyplot
import pyqtgraph.opengl as gl

def generate_surface(func, xmin, xmax, ymin, ymax, npoints=300):
    x = np.linspace(xmin, xmax, npoints)
    y = np.linspace(ymin, ymax, npoints)
    X, Y = np.meshgrid(x, y)
    Z_raw = func(X,Y)
    Z_min = Z_raw.min()
    Z_max = Z_raw.max()
    if Z_max != Z_min:
        Z = (Z_raw - Z_min) / (Z_max - Z_min) * 10
    else:
        Z = Z_raw

    Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
    cmap = pyplot.get_cmap("jet")
    colors = cmap(Z_norm)
    surface_item = gl.GLSurfacePlotItem(
        x=x,
        y=y,
        z=Z,
        colors=colors,
        smooth=True,
        drawEdges=False,
        shader="shaded"
    )
    return surface_item, Z