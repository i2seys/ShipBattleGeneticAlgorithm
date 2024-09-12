from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

v_ship = np.array([0, 1, 2, 3])
h_ship = np.array([0, 0, 0, 0])
type_ship = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]
colors = ['g', 'b', 'm', 'y']

inf = 100

def show_ships(ax, best, field_size):
    # Отвечает за красный фон у поля
    rect = Rectangle((0, 0), field_size + 1, field_size + 1, fill=None, edgecolor='r')

    # Рисует линии у поля
    for i in range(field_size + 1):
        ax.add_line(Line2D((i + 0.5, i + 0.5), (0 + 0.5, field_size+ 0.5), color='#aaa'))
        ax.add_line(Line2D((0 + 0.5, field_size + 0.5), (i + 0.5, i + 0.5), color='#aaa'))

    t_n = 0
    # Отрисовывает корабли
    for i in range(0, len(best), 3):
        x = best[i]
        y = best[i + 1]
        r = best[i + 2]
        t = type_ship[t_n]
        t_n += 1

        if r == 1:
            ax.plot(v_ship[:t] + x, h_ship[:t] + y, ' sb', markersize=18, alpha=0.8, markerfacecolor=colors[t - 1])
        else:
            ax.plot(h_ship[:t] + x, v_ship[:t] + y, ' sb', markersize=18, alpha=0.8, markerfacecolor=colors[t - 1])

    ax.add_patch(rect)