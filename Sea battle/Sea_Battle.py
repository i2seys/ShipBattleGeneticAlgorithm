import random
import matplotlib.pyplot as plt

import numpy as np

from deap import base
from deap import creator
from deap import tools

from Get_Elitizme import eaSimpleElitizme
from Show_ships import show_ships

# Размер поля - 10x10
FIELD_SIZE = 10
# Сколько кораблей - 10
SHIPS = 10
# Длина хромосомы
# Хромосома представляет собой 3 цифры:
# Первая - координата X, вторая - координата Y, третья - куда смотрит корабль (вертикально или горизонтально)
# Длина корабля определяется началом его гена в хромосоме, то есть первые 3 числа особи описывают корабль длины 4 и т.д:
# 444 | 333 333 | 222 222 222 | 111 111 111 111
LENGHT_CHROM = 3 * SHIPS

# Размер популяции
POPULATION_SIZE = 500
# Вероятность скрещивания
P_CROSSOVER = 0.9
# Вероятность мутации
P_MUTATION = 0.3
# Максимальное количество эволюций
MAX_GENERATIONS = 50
# Количество лучших особей для вывода
HALL_OF_FAME_SIZE = 1

# Описывает клетку вне поля
inf = 100
# Штраф при наложении
imposition_penalpy = 200
# Штраф при пересечении границы
cross_boarder_penalty = 50
# Описывает границы вокруг корабля
boarder_fine = 1
# Описывает клетку корабля
ship_fine = 10

# Функция отображения новой расстановки кораблей в окне
# Вызывается после каждого нового поколения
def show(ax, hof):
	ax.clear()
	plt.xticks(np.arange(1, 11, 1))
	plt.yticks(np.arange(1, 11, 1))
	show_ships(ax, hof.items[0], FIELD_SIZE)

	plt.draw()
	plt.gcf().canvas.flush_events()


# Создаёт случайные корабли (количеством total - указывается 10 то есть количество кораблей)
def randomShip(total):
	ships = []
	for n in range(total):
		# Координаты и направление
		ships.extend([random.randint(1, FIELD_SIZE), random.randint(1, FIELD_SIZE), random.randint(0, 1)])

	return creator.Individual(ships)


# Функция вычисления приспособленности одного индивидуума.
# Сам по себе критерий приспособленности:
# - правильно расставленные корабли (без нарушения правил)
# - как можно меньшая вероятность подбить корабль (чем меньше полей будут являться границами и и кораблями - тем лучше)
def shipsFitness(individual):
	type_ship = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]
	P0 = np.zeros((FIELD_SIZE, FIELD_SIZE)) # Игровое поле для кораблей
	P = np.ones((FIELD_SIZE + 6, FIELD_SIZE + 6)) * inf # Игровое поле для кораблей с учетом выхода за границы
	P[1:FIELD_SIZE + 1, 1:FIELD_SIZE + 1] = P0

	# Границы корабля в рамках поля. Для вертикального корабля с длиной 2 границы будут такими:
	# 1 1  1
	# 1 10 1
	# 1 10 1
	# 1 1  1
	h = np.ones((3, 6)) * boarder_fine
	ship_one = np.ones((1, 4)) * ship_fine
	v = np.ones((6, 3)) * boarder_fine

	# Корабли рисуются на поле (они могут даже наложиться друг на друга)
	# Как: по порядку перебираются все корабли (4, 33, 222, 1111) и каждый из них наносится на поле.
	# Точка с кораблём имеет значение 10, точка с границей корабля - 1.
	for *ship, t in zip(*[iter(individual)] * 3, type_ship):
		if ship[-1] == 0:
			sh = np.copy(h[:, :t + 2])
			sh[1, 1:t + 1] = ship_one[0, :t]
			P[ship[0] - 1:ship[0] + 2, ship[1] - 1:ship[1] + t + 1] += sh
		else:
			sh = np.copy(v[:t + 2, :])
			sh[1:t + 1, 1] = ship_one[0, :t]
			P[ship[0] - 1:ship[0] + t + 1, ship[1] - 1:ship[1] + 2] += sh

	# for i in range(len(P)):
	# 	for j in range(len(P[i])):
	# 		if P[i][j] > 101 and i > 12 and j > 13:
	# 			P[i][j] = 100
	# Корректировка игрового поля и выдача штрафов
	for i in range(len(P)):
		for j in range(len(P[i])):
			# Если данное поле является клеткой, на которой нет корабля и клеткой внутри игрового поля (не снаружи)
			# то его значение приравнивается к клетке, на которой нет корабля - то есть к 1.
			if P[i][j] > 0 and P[i][j] < ship_fine:
				P[i][j] = boarder_fine
			# Если в клетке более 2 кораблей и клетка игровая (то есть не снаружи поля)
			# То этой клетке выдаётся штраф (+200)
			if P[i][j] >= 2 * ship_fine and i != 0 and j != 0 and i <= 10 and j <= 10:
				P[i][j] += imposition_penalpy
			# Если данная клетка говорит о том, что рядом есть ещё какой-то корабль
			# (его граница наслаивается на границу текущего корабля),
			# то за это тоже присваивается штраф
			if P[i][j] % ship_fine != 0 and P[i][j] >= ship_fine and i != 0 and j != 0 and i <= 10 and j <= 10:
				P[i][j] += cross_boarder_penalty
			# Если клетка вне поля и в ней всё в порядке (нет выхода корабля за границу),
			# то она не учитывается при подсчёте
			if (i == 0 or j == 0 or i > 10 or j > 10) and (P[i][j] < inf + ship_fine):
				P[i][j] = inf


	s = 0
	# Подсчитывание общего количества клеток, занятых кораблями и их границами
	for i in range(len(P)):
		for j in range(len(P[i])):
			# Если клетка находится в границе игрового поля, то её суммируем
			if i != 0 and j != 0 and i <= 10 and j <= 10:
				s += P[i][j]
			else:
				# Если клетка вне игрового поля, но в ней есть ошибка, то суммируем и ошибку
				if P[i][j] >= inf + ship_fine:
					s += P[i][j]

	return s,

# Мутация. indpb - вероятность мутации.
# В данном случае при мутации перебираются все гены хромосомы особи и для каждого гена с определённой
# вероятностью происходят изменения. Если ген отвечает за ориентацию (горизонтальную или вертикальную),
# то он меняется в пределах от 0 до 1.
# Если же ген отвечает за координаты расположения на поле, то он изменяется в пределах от 1 до длины поля.
def MutShip(individual, indpb):
	for i in range(len(individual)):
		if random.random() < indpb:
			individual[i] = random.randint(0, 1) if (i + 1) % 3 == 0 else random.randint(1, FIELD_SIZE)

	return individual,


def __main__():
	hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

	# Создаёт фабрику по созданию особи.
	# Название класса - Individual, базовый класс - list, аргументы для создания = функция fitnessMin
	# В данном случае fitness и weights являются атрибутами экземпляра.
	# После этого можно создавать экземпляры Individual с помощью creator.Individual()
	# Для оценки особи будет использоваться одно число (1.0), и его нужно минимизировать (-1.0)
	# То есть будет происходить поиск минимум функции приспособленности
	# Формула вычисления приспособленности: (val[0]*weight[0],val[1]*weight[1],...)
	creator.create("FitnessMin", base.Fitness, weights=(-1.0, ) )
	creator.create("Individual", list, fitness=creator.FitnessMin)

	# Определение вспомогательных функций с помощью toolbox. Указывается имя функции, функция и аргументы
	toolbox = base.Toolbox()
	# Функция создания случайной особи
	toolbox.register("randomShip", randomShip, SHIPS)
	# Функция создания популяции
	# название - populationCreator, "скелет" для создания функции - tools.initRepeat,
	# далее идут её аргументы -
	# 	контейнер для хранения генов (list),
	# 	функция генерации значения гена (toolbox.randomShip),
	# 	число генов в хромосоме (указана ранее)
	toolbox.register("populationCreator", tools.initRepeat, list, toolbox.randomShip)

	# Задаём функцию вычисления приспособленности индивидуума
	toolbox.register("evaluate", shipsFitness)
	# Задаём функцию отбора (селекции) - она будет основана на турнирном отборе из трёх участников.
	toolbox.register("select", tools.selTournament, tournsize=3)
	# Задаём функцию скрещивания. Выполняет двухточечное скрещивание
	toolbox.register("mate", tools.cxTwoPoint)
	# Задаём функцию для мутации. Мутацию задали собственноручно. Её вероятность - 1.0/ длина хромосомы (то есть 1/30)
	toolbox.register("mutate", MutShip, indpb=1.0 / LENGHT_CHROM)

	# Сохраняет в себе статистику каждого поколения.
	# Номер эволюции, количество некорректных особей, минимальную и среднюю приспособленность
	# Строчка ниже описывает, какие данные индивидуума будут браться для вычисления статистики
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("min", np.min)
	stats.register("avg", np.mean)

	# Графики
	plt.ion()
	fig, ax = plt.subplots()
	fig.set_size_inches(6, 6)

	ax.set_xlim(-2, FIELD_SIZE + 3)
	ax.set_ylim(-2, FIELD_SIZE + 3)

	# Создание популяции
	population = toolbox.populationCreator(n=POPULATION_SIZE)

	# Запуск алгоритма и сохранение его результата (популяции и информации об эволюции) в переменные
	population, logbook = eaSimpleElitizme(
		population, toolbox, cxpb=P_CROSSOVER,
		mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
		halloffame=hof, stats=stats,
		callback=(show, (ax, hof, )), verbose=True
	)

	# Получение лучшей особи за всё время
	best = hof.items[0]
	print(best)

	plt.ioff()
	plt.show()

	# Вывод графика зависимости приспособленности от поколений
	maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")
	plt.plot(maxFitnessValues, color='red')
	plt.plot(meanFitnessValues, color='green')
	plt.xlabel("Generation")
	plt.ylabel("Max/avg fitness")
	plt.show()


if __name__ == '__main__':
	__main__()