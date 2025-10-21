import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

c = 340

def velocity(h, tau, c):
    return c / np.sqrt(1 - (c*tau/h)**2)

table = PrettyTable()
table.field_names = ["№","h (м)","tau (с)","v (м/с)"]
table.float_format = '.3'

h_values = np.arange(500, 5001, 500)
t_values = np.arange(2, 15, 2)
count = 1

for h in h_values:
    for t in t_values:
        if h <= c * t:
            continue
        v = velocity(h, t, c)
        table.add_row([count, h, t, v])
        count += 1

print(table)

#h змінюється, t стале
t_values = [2,5,8,11,14]
h_values = np.linspace(500, 5001, 500)

for t in t_values:
    h_valid = h_values[h_values > c * t * 1.01]
    v = velocity(h_valid, t, c)
    plt.plot(h_valid, v, label=f"t = {t} с")
plt.xlabel("Висота (м)")
plt.ylabel("Швидкість (м/с)")
plt.title("Залежність швидкості від висоти при сталому часі.")
plt.legend()
plt.show()

#t змінюється, h стале
h_values = [500,1000,2000,3000,4500]
t_values = np.linspace(0.01, 15, 500)

for h in h_values:
    t_valid = t_values[t_values < h / c * 0.99]
    v = velocity(h, t_valid, c)
    plt.plot(t_valid, v, label=f"h = {h} м")
plt.xlabel("Час затримки (c)")
plt.ylabel("Швидкість (м/с)")
plt.title("Залежність швидкості від часу затримки при сталій висоті.")
plt.legend()
plt.show()

h = np.linspace(500, 5000, 200)
tau = np.linspace(0.01, 15, 200)
H, T = np.meshgrid(h, tau)

# Обчислюємо швидкість (тільки в області tau < h/c)
V = velocity(H, T, c)
V = np.where(T < H/c-1, V, np.nan)

# Створення 3D-графіка
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_wireframe(H, T, V, rstride=20, cstride=20)
ax.set_xlabel("Висота h, м")
ax.set_ylabel("Затримка τ, с")
ax.set_zlabel("Швидкість v, м/с")
ax.set_title("3Д-графік залежності швидкості літака")
plt.show()