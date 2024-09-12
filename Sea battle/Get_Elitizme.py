from deap import tools
from deap.algorithms import varAnd


def eaSimpleElitizme(population, toolbox, cxpb, mutpb, ngen, stats=None,
                     halloffame=None, verbose=__debug__, callback=None):
    
    # Переменная для хранения информации о поколениях
    logbook = tools.Logbook()
    logbook.header = ['gen'] + (stats.fields if stats else [])

    # Подсчёт приспособленности каждой особи
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Добавление лучшей особи в переменную для хранения лучшей особи
    if halloffame is not None:
        halloffame.update(population)

    # Средняя и лучшая приспособленнось поколения
    record = stats.compile(population) if stats else {}
    # Сохранение текущей популяции в logbook.
    logbook.record(gen=0, **record)
    if verbose:
        print(logbook.stream)

    # Начало алгоритма. Вычисление первой популяции (не считая исходной)
    for gen in range(1, ngen + 1):
        # Отбор (Турнирный)
        # второй аргумент - k - для турнирного отбора это количество особей для отбора.
        offspring = toolbox.select(population, len(population))

        # Скрещивание и мутация
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # Вычисляется приспособленность
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Обновление лучшей особи
        offspring.extend(halloffame.items)

        if halloffame is not None:
            halloffame.update(offspring)

        # Обновление популяции
        population[:] = offspring

        # Сохранение статистических данных
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, **record)
        if verbose:
            print(logbook.stream)

        if callback:
            callback[0](*callback[1])

    return population, logbook

    