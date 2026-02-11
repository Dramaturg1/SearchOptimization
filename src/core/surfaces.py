import numpy as np

def beale(X, Y):
    return (1.5 - X + X * Y) ** 2 + (2.25 - X + X * Y ** 2) ** 2 + (2.625 - X + X * Y ** 3) ** 2

def booth(X, Y):
    return (X + 2*Y - 7)**2 + (2*X + Y - 5)**2

surface_functions = {
    "Функция Била": beale,
    "Функция Бута": booth
}