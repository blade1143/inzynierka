import random
import collections
import string
import matplotlib.pyplot as plt


# przedmioty, wagi, wartosci
def items_list():
    items = [
        ("map", 9, 150),
        ("compass", 13, 35),
        ("water", 153, 200),
        ("water", 153, 200),
        ("water", 153, 200),
        ("sandwich", 50, 60),
        ("sandwich", 50, 60),
        ("glucose", 15, 60),
        ("glucose", 15, 60),
        ("tin", 68, 45),
        ("tin", 68, 45),
        ("tin", 68, 45),
        ("banana", 27, 60),
        ("banana", 27, 60),
        ("banana", 27, 60),
        ("apple", 39, 40),
        ("apple", 39, 40),
        ("apple", 39, 40),
        ("cheese", 23, 30),
        ("beer", 52, 10),
        ("beer", 52, 10),
        ("beer", 52, 10),
        ("suntan cream", 11, 70),
        ("camera", 32, 30),
        ("t-shirt", 24, 15),
        ("t-shirt", 24, 15),
        ("trousers", 48, 10),
        ("trousers", 48, 10),
        ("umbrella", 73, 40),
        ("waterproof trousers", 42, 70),
        ("waterproof overclothes", 43, 75),
        ("note-case", 22, 80),
        ("sunglasses", 7, 20),
        ("towel", 18, 12),
        ("towel", 18, 12),
        ("socks", 4, 50)
    ]
    random.shuffle(items)
    list_of_weight = [weight[1] for weight in items]
    list_of_value = [value[2] for value in items]

    return items, list_of_weight, list_of_value


def random_string(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def items_generator(items_quantity, min_w, max_w, min_v, max_v, max_it):
    items_list = []
    for i in range(items_quantity):
        random_name = random_string()
        random_weight = random.randint(min_w, max_w)
        random_value = random.randint(min_v, max_v)
        random_quantity = random.randint(1, max_it)

        for j in range(random_quantity):
            items_list.append((random_name, random_weight, random_value))

    list_of_weight = [weight[1] for weight in items_list]
    list_of_value = [value[2] for value in items_list]

    return items_list, list_of_weight, list_of_value


def random_chromosome(len_items):
    chromosome = [random.randint(0, 1) for each_item in range(len_items)]
    return chromosome


def checking_backpack(given_chromosome):
    '''
    wagi, wartosci
    :param given_chromosome: chromosom losowy z pakowanymi przedmiotami
    :return: zwraca mi wage i wartosc chromosomu
    '''
    sum_of_weight = 0
    sum_of_value = 0
    for index in range(len(given_chromosome)):
        if given_chromosome[index] == 1:
            sum_of_weight += weights[index]
            sum_of_value += values[index]
    return [given_chromosome, [sum_of_weight, sum_of_value]]


def random_population(size_population, items_size):
    '''
    :param size_population: ilosc populacji do algorytmu
    :return: pelna populacje z obliczonymi wagami i wartosciami
    '''
    len_of_items = items_size
    list_of_full_population = [checking_backpack(random_chromosome(len_of_items)) for each_element in
                               range(size_population)]

    return list_of_full_population


def fitness_function(population, max_weight):
    '''
    tworzy liste przedmiotów które są wstanie zmiescic sie w plecaku
    :param population:
    :param max_weight:
    :return:
    '''
    list_of_full_population = population[:]
    population_lower_than_weights = []
    for index in range(len(list_of_full_population)):
        if list_of_full_population[index][1][0] <= max_weight:
            population_lower_than_weights.append(list_of_full_population[index])

    # x = sorted(population_lower_than_weights, key=lambda by_value: by_value[1][1], reverse=True)
    x = population_lower_than_weights

    return x


def fitness_function_modified(population, max_weight):
    '''
    tworzy liste przedmiotów które są wstanie zmiescic sie w plecaku
    :param population:
    :param max_weight:
    :return:
    '''
    list_of_full_population = population[:]
    population_lower_than_weights = []
    for index in range(len(list_of_full_population)):
        if list_of_full_population[index][1][0] <= max_weight:
            population_lower_than_weights.append(list_of_full_population[index])

    # x = sorted(population_lower_than_weights, key=lambda by_value: by_value[1][1], reverse=True)
    x = population_lower_than_weights

    return x


def selection_stage_one(population):
    '''
    wybor 2/5 zbioru losowo
    :param n:
    :return: zwraca 2/5 zbioru oraz pozostala populacje do kolejnego etapu
    '''
    fitness_population = population[:]
    sorted_fitness_pop = sorted(fitness_population, key=lambda by_value: by_value[1][1], reverse=True)
    stage_one = round(len(fitness_population) * (2 / 5))
    chromosome_stage_one = sorted_fitness_pop[:stage_one]
    # print(chromosome_stage_one)
    remaining_population = sorted_fitness_pop[stage_one:]

    return chromosome_stage_one, remaining_population


def selection_stage_two(remaining_population):
    '''
    zwraca chromosomy stage one i stage two czyli 33% z 3/5 pozostalej populacji
    :param population:
    :return:
    '''
    remaining_population_second = remaining_population[:]
    # stage_two = round(len(remaining_population_second) * (33 / 100))
    # chromosome_stage_two = remaining_population_second[:stage_two]
    # remaining_population_second = remaining_population_second[stage_two:]
    chromosome_stage_two = random.sample(remaining_population_second,
    									 k=int(len(remaining_population_second) * (33 / 100)))
    [remaining_population_second.remove(chromosome_stage_two[i]) for i in range(len(chromosome_stage_two))]

    return chromosome_stage_two, remaining_population_second


'''
wycofujemy metode koła ruletki w wyborze osobnikow do krzyzowania
'''


# def selection_stage_three(remaining_population_second):
# 	'''
# 	wybieranie metoda kola ruletki osobniki do krzyzowania
# 	:param population:
# 	:return: liste chromosomow ktore zostaly wybrane (moga sie powtarzac)
# 	'''
# 	# random.shuffle(remaining_population_second)   #  moze sie przydac
# 	roulete_wheel_sum = sum([(each_elem[1][1]) for each_elem in remaining_population_second])
# 	percentage_roulete = []
# 	percentage_roulete_2 = [0]
#
# 	for each_elem in remaining_population_second:
# 		percentage_roulete.append(round((each_elem[1][1] / roulete_wheel_sum) * 1000, 2))
#
# 	for i in range(len(percentage_roulete)):
# 		percentage_roulete_2.append(percentage_roulete_2[i] + percentage_roulete[i])
# 	percentage_roulete_2.remove(0)
#
# 	random_numbers = random.sample(range(0, 1000), k=len(remaining_population_second))
#
# 	choosed_chromosome_list = []
# 	for index_i in range(len(random_numbers)):
# 		for index_j in range(len(percentage_roulete_2)):
# 			if random_numbers[index_i] < percentage_roulete_2[index_j]:
# 				choosed_chromosome_list.append(remaining_population_second[index_j])
# 				break
#
# 	return choosed_chromosome_list


def crossing_version_1(x, y):
    '''
    pobranie rodzicow i skrzyzowanie dla powstania potomkow
    :param x: chromosom wylosowany do krzyzowania
    :param y: para do krzyzowania dla chromosomu x
    :return: skrzyzowany nowy potomek
    '''
    cross_point = random.randint(1, len(x[0]) - 1)
    crossed = x[0][:cross_point] + y[0][cross_point:]

    return crossed


def crossover_v1(list_of_chromosome, crossover_chance):
    '''
    pobieramy wyselekcjonowane chromosomy do krzyzowania
    losujemy dla kazdego chromosomu czy bedzie sie krzyzowal
    dodajemy do osobnej listy te ktore beda sie krzyzowac, nastepnie losujemy z calej populacji dla nich pary
    krzyzujemy i zastepujemy rodzicow nowymi potomkami
    ostatecznie obliczamy wagi i wartosci dla nowych potomkow i zwracamy skrzyzowane chromosomy
    :param n:
    :return:
    '''
    list_of_chromosome_copy = list_of_chromosome[:]
    crossover_chance_0_1 = [1 if random.random() < crossover_chance else 0 for i in range(len(list_of_chromosome_copy))]

    new_list_to_crossover = [list_of_chromosome_copy[index] for index in range(len(list_of_chromosome_copy)) if
                             crossover_chance_0_1[index] == 1]

    list_of_crossover_pairs = random.choices(list_of_chromosome_copy, k=len(new_list_to_crossover))

    [list_of_chromosome_copy.remove(new_list_to_crossover[i]) for i in range(len(new_list_to_crossover))]

    finally_list_crossed = [
        crossing_version_1(new_list_to_crossover[each_chromosome], list_of_crossover_pairs[each_chromosome])
        for each_chromosome in range(len(new_list_to_crossover))]

    finally_list_crossed_calculated = [checking_backpack(each_chromosome) for each_chromosome in finally_list_crossed]

    list_of_chromosome_copy.extend(finally_list_crossed_calculated)

    return list_of_chromosome_copy


def mutating(chromosome, mutation_chance):
    '''
    mutowanie genów
    :param chromosome:
    :param mutation_chance:
    :return:
    '''
    chromosome_copy = chromosome[:]
    mutated_chromosome = []
    for i in chromosome_copy[0]:
        mutated_gen = random.random()
        if (mutated_gen < mutation_chance) and i == 0:
            mutated_chromosome.append(i + 1)
        elif (mutated_gen < mutation_chance) and i == 1:
            mutated_chromosome.append(i - 1)
        else:
            mutated_chromosome.append(i)
    return mutated_chromosome


def mutations(population, mutation_chance):
    '''
    szansa na mutowanie i wrzucanie nowych mutacji do ostatecznej listy
    :param population:
    :param crossover_chance:
    :return:
    '''
    population_copy = population[:]
    finally_mutated_list = [mutating(each_chromosome, mutation_chance) for each_chromosome in population_copy]

    finally_list_mutated_calculated = [checking_backpack(each_chromosome) for each_chromosome in finally_mutated_list]

    return finally_list_mutated_calculated


def matching_items(result_items, items):
    '''
    funkcja dobierajaca w pary przedmioty spakowane i przedstawiajaca ilosci przedmiotow
    :param result_items:
    :param items:
    :return:
    '''
    list_of_items = []
    for i in range(len(result_items[0])):
        for j in range(len(items)):
            if result_items[0][j] == 1:
                list_of_items.append(items[j][0])
        list_of_items.append((['waga:', result_items[1][0], 'wartosc:', result_items[1][1]]))
        break
    return list_of_items


def counting_items(data):
    '''
    liczenie przedmiotow spakowanych
    :param data:
    :return:
    '''
    result = collections.Counter(data[:-1])

    return result, data[-1]


def crossing_version_2(to_cross):
    '''
    pobranie rodzicow i skrzyzowanie dla powstania potomkow
    :param x: chromosom wylosowany do krzyzowania
    :param y: para do krzyzowania dla chromosomu x
    :return: skrzyzowany nowy potomek
    '''
    to_cross_copy = to_cross[:]
    children_1 = []
    crossed_chromosome = []

    for elem in to_cross_copy:
        for j in range(len(elem[0])):
            children_1.append(elem[0][j])
            children_1.append(elem[1][j])
        children_1 = children_1[::-1]
        children_2 = children_1[:int(len(children_1) / 2)]
        children_1 = children_1[int(len(children_1) / 2):]
        crossed_chromosome.append(children_2)
        crossed_chromosome.append(children_1)

        children_1, children_2 = [], []

    return crossed_chromosome


def crossover_version_2(population_to_cross, crossover_chance):
    '''
    losuje pary do krzyzowania z ostatniej selekcji, dla X prawdopobienstwa sprawdza czy Y para sie krzyzuje,
    krzyzuje osobniki w sposob:
    r1 = [1 1 1 1], r2 = [0 0 0 0]
    c1 = [1 0 1 0], c2 = [1 0 1 0]
    jezeli krzyzowanie nie zwroci chromosomow ktore sa dopuszczalne, powtarzam procedure od momentu losowania w pary
    :param population_to_cross:
    :param crossover_chance:
    :return:
    '''

    population_to_cross_copy = population_to_cross[:]

    random.shuffle(population_to_cross_copy)
    elem_for_last = 0
    if len(population_to_cross_copy) % 2 != 0:
        elem_for_last = population_to_cross_copy.pop()

    pairs_to_cross = []
    pairs_without_calculate = []
    for i in range(len(population_to_cross_copy)):
        pairs_without_calculate.append(population_to_cross_copy[i][0])

    random.shuffle(pairs_without_calculate)
    while pairs_without_calculate != []:
        pairs_to_cross.append([pairs_without_calculate.pop(), pairs_without_calculate.pop()])

    crossover_chance_0_1 = [1 if random.random() < crossover_chance else 0 for i in range(len(pairs_to_cross))]

    pairs_to_cross_chance = []
    for index in range(len(pairs_to_cross)):
        if crossover_chance_0_1[index] == 1:
            pairs_to_cross_chance.append(pairs_to_cross[index])

    for id in pairs_to_cross_chance:
        pairs_to_cross.remove(id)

    back_from_pair = []
    for i in pairs_to_cross:
        back_from_pair.extend(i)

    crossed_chromos = crossing_version_2(pairs_to_cross_chance)

    if elem_for_last != 0:
        connect = crossed_chromos + back_from_pair + [elem_for_last[0]]
    else:
        connect = crossed_chromos + back_from_pair

    finally_connect_with_calc = [checking_backpack(item) for item in connect]

    return finally_connect_with_calc


def results_making(return_of_main, max_weight_given):
    '''
    funkcja pobiera ostateczny wynik algorytmu, pokazuje ilosc spakowanych przedmiotow + wartosc i wage chromosomu
    :param return_of_main:
    :param max_weight_given:
    :return:
    '''
    tab_for_our_limit = []
    for each_elem in return_of_main:
        elem = fitness_function(each_elem, max_weight_given)
        if elem != []:
            tab_for_our_limit.extend(elem)

    best_chromosome = sorted(tab_for_our_limit, key=lambda by_value: by_value[1][1], reverse=True)[0]
    best_with_names = matching_items(best_chromosome, items_lists[:len(best_chromosome[0])])
    finally_items = counting_items(best_with_names)

    print(f'ilosc iteracji po ktorych mamy wynik: {iters + 1} \n'
          f'ilość populacji: {population_} \n'
          f'maksymalna waga przedmiotow to: {sum(weights)} \n'
          f'dopuszczalna waga: {max_weight_given} \n'
          f'maksymalna wartosc: {sum(values)} \n')

    return finally_items


def main(iterations, population, mutation_chance, crossover_chance, max_weight, items_size):
    '''
    calkowite uruchomienie algorytmu
    :return: liste solucji wszystkich do kolejnego etapu
    '''
    result = []
    second = random_population(population, items_size)
    third = fitness_function(second, max_weight)
    result.append(third)

    for i in range(iterations):
        # fourth + sixth + nineth
        fourth, fifth = selection_stage_one(third)
        sixth, seventh = selection_stage_two(fifth)
        nineth = crossover_version_2(seventh, crossover_chance)

        while len(nineth) < len(seventh):
            ten = crossover_version_2(seventh, crossover_chance)
            nineth = ten

        connected_selection = nineth + sixth + fourth
        random.shuffle(connected_selection)
        tenth = mutations(connected_selection, mutation_chance)
        third = fitness_function(tenth, max_weight)

        if third == []:
            break
        result.append(third)

    return result, i


global items_lists, weights, values
# items_lists, weights, values = items_generator(items_quantity=15, min_w=10, max_w=50, min_v=10, max_v=150, max_it=4)
items_lists, weights, values = items_list()
size_of_items_quantity = len(items_lists)

print(f'ilosc przedmiotow w chromosomie: {size_of_items_quantity}')
z = 0
epocs = 1
while z < epocs:
    weight_to_set = 400
    population_ = 10000
    something, iters = main(iterations=100, population=population_, mutation_chance=0.05,
                            crossover_chance=0.5, max_weight=weight_to_set, items_size=size_of_items_quantity)
    try:
        ss = results_making(something, weight_to_set)
        print(ss[0], '\n', ss[1])
    # value_of_ss = ss[1][3]
    # d = sum(values)
    # a = ( (d - value_of_ss) / d ) * 100
    # print('\nbłąd procentowy od maxymalnej wartości rozwiązania:', a, '\n\n')
    except:
        print('nie znaleziono rozwiązań')
    z += 1


def plot_making(data):
    w, v, it = [], [], []
    ii = 0
    for i in data:
        sorted_vector = sorted(i, key=lambda by_value: by_value[1][1], reverse=True)
        w.append(sorted_vector[0][1][0])
        v.append(sorted_vector[0][1][1])
        it.append(ii)
        ii += 1

    plt.figure(figsize=(18, 8))
    plt.grid()
    plt.plot(it, w, label='waga')
    plt.plot(it, v, label='wartość')
    plt.plot(v.index(max(v)), max(v), '*', label='wartość maksymalna')
    plt.title('Najlepszy chromosom z populacji w poszczególnej iteracji')
    plt.xlabel('Liczba iteracji')
    plt.ylabel('Wartość zapakowanych przedmiotów')
    plt.legend()
    plt.show()


plot_making(something)
