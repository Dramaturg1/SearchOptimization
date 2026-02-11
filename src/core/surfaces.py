import numpy as np

def beale(X, Y):
    return (1.5 - X + X * Y) ** 2 + (2.25 - X + X * Y ** 2) ** 2 + (2.625 - X + X * Y ** 3) ** 2

def booth(X, Y):
    return (X + 2*Y - 7)**2 + (2*X + Y - 5)**2

def ackley(X,Y):
    return -20*np.exp(-0.2*np.sqrt(0.5*(X*X + Y*Y))) - np.exp(0.5*(np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))) + np.exp(1) + 20

def goldstein_price(X,Y):
    return (1 + (X+Y+1)**2*(19-14*X+3*X**2-14*Y+6*X*Y+3*Y**2))*(30+(2*X-3*Y)**2*(18-32*X+12*X**2+48*Y-36*X*Y+27*Y**2))

surface_functions = {
    "Функция Била": beale,
    "Функция Бута": booth,
    "Функция Экли": ackley,
    "Функция Гольдшейна-Прайса": goldstein_price
}