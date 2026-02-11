import numpy as np

def beale(X, Y):
    return (1.5 - X + X * Y) ** 2 + (2.25 - X + X * Y ** 2) ** 2 + (2.625 - X + X * Y ** 3) ** 2

def booth(X, Y):
    return (X + 2*Y - 7)**2 + (2*X + Y - 5)**2

def ackley(X,Y):
    return -20*np.exp(-0.2*np.sqrt(0.5*(X*X + Y*Y))) - np.exp(0.5*(np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))) + np.exp(1) + 20

def goldstein_price(X,Y):
    return (1 + (X+Y+1)**2*(19-14*X+3*X**2-14*Y+6*X*Y+3*Y**2))*(30+(2*X-3*Y)**2*(18-32*X+12*X**2+48*Y-36*X*Y+27*Y**2))

def bukin_n6(X,Y):
    return 100*np.sqrt(np.abs(Y-0.01*X**2)) + 0.01*np.abs(X+10)

def matyas(X,Y):
    return 0.26*(X**2+Y**2) - 0.48*X*Y

def levi_n13(X,Y):
    return (np.sin(3*np.pi*X))**2 + (X-1)**2*(1+np.sin(3*np.pi*X))+(Y-1)**2*(1+np.sin(2*np.pi*Y)**2)

def himmelblau(X,Y):
    return (X**2+Y-11)**2 + (X + Y**2 - 7)**2

def three_hump_camel(X,Y):
    return 2*X**2 - 1.05*X**4 + X**6/6 + X*Y + Y**2

def easom(X,Y):
    return -np.cos(X)*np.cos(Y)*np.exp(-((X-np.pi)**2+(Y-np.pi)**2))

def cross_in_tray(X, Y):
    return -0.0001*(np.abs(np.sin(X)*np.sin(Y)*np.exp(np.abs(100-(np.sqrt(X*X+Y*Y))/np.pi)))+1)**0.1

def eggholder(X,Y):
    return -(Y+47)*np.sin(np.sqrt(np.abs(X/2+(Y+47)))) - X*np.sin(np.sqrt(np.abs(X-(Y+47))))

def mccormick(X,Y):
    return np.sin(X+Y)+(X-Y)**2-1.5*X+2.5*Y+1

def schaffer_n2(X,Y):
    return 0.5+((np.sin(X*X-Y*Y))**2-0.5)/(1+0.001*(X*X+Y*Y))**2

def schaffer_n4(X,Y):
    return 0.5+(np.cos((np.sin(np.abs(X*X-Y*Y))))**2-0.5)/(1+0.001*(X*X+Y*Y))**2

def holder(X,Y):
    return -np.abs(np.sin(X)*np.cos(Y)*np.exp(np.abs(1-np.sqrt(X*X+Y*Y)/np.pi)))

surface_functions = {
    "Функция Била": beale,
    "Функция Бута": booth,
    "Функция Экли": ackley,
    "Функция Гольдшейна-Прайса": goldstein_price,
    "Функция Букина N 6": bukin_n6,
    "Функция Матьяса": matyas,
    "Функция Леви N 13": levi_n13,
    "Функция Химмельблау": himmelblau,
    "Функция трёхгорбого верблюда": three_hump_camel,
    "Функция Изома": easom,
    "Функция \"крест на подносе\"": cross_in_tray,
    "Функция \"подставка для яиц\"": eggholder,
    "Функция МакКормика": mccormick,
    "Функция Шаффера N2": schaffer_n2,
    "Функция Шаффера N4": schaffer_n4,
    "Табличная функция Хольдера": holder
}