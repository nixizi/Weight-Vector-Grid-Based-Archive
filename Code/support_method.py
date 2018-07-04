import math
import numpy as np
import copy
import random
import decomposition_method
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
import sys
import pygmo as pg
import time
import pandas as pd
from pandas.plotting import parallel_coordinates
import problem_set
import operator
import functools


def update_archive(weight_vectors, archive, new_solution, refer_vector, update_type="shuffle"):
    if update_type == "archive":
        update_archive_origin(weight_vectors, archive,
                              new_solution, refer_vector)
    elif update_type == "shuffle":
        update_archive_shuffle(weight_vectors, archive,
                               new_solution, refer_vector)
    elif update_type == "random":
        update_archive_random(weight_vectors, archive,
                              new_solution, refer_vector)
    else:
        pass


def update_archive_origin(weight_vectors, archive, new_solution, refer_vector):
    minimum_angle = sys.maxsize
    nearby_vector_index = None

    # Find the closest vectors
    for i in range(len(weight_vectors)):
        wv = weight_vectors[i, :]
        cos_value = (np.dot(wv, new_solution)) / \
            math.sqrt((wv.dot(wv.T)) * (new_solution.dot(new_solution.T)))
        angle = math.acos(cos_value)
        if angle < minimum_angle:
            nearby_vector_index = i
            minimum_angle = angle
    nearby_vector = weight_vectors[nearby_vector_index]

    # Update archive
    if nearby_vector_index not in archive:
        archive[nearby_vector_index] = new_solution
    else:
        original_solution = archive[nearby_vector_index]
        if decomposition(new_solution, nearby_vector, refer_vector, 5) < decomposition(original_solution, nearby_vector, refer_vector, 5):
            archive[nearby_vector_index] = new_solution


def update_archive_shuffle(weight_vectors, archive, new_solution, refer_vector):
    minimum_angle = sys.maxsize
    nearby_vector_index = None
    index_pool = [x for x in range(len(weight_vectors))]
    random.shuffle(index_pool)

    # Find the closest vectors
    for i in index_pool:
        wv = weight_vectors[i, :]
        cos_value = (np.dot(wv, new_solution)) / \
            math.sqrt((wv.dot(wv.T)) * (new_solution.dot(new_solution.T)))
        angle = math.acos(cos_value)
        if angle < 0.01:
            minimum_angle = angle
            nearby_vector_index = i
            break
        if angle < minimum_angle:
            nearby_vector_index = i
            minimum_angle = angle
    nearby_vector = weight_vectors[nearby_vector_index]

    # Update archive
    if nearby_vector_index not in archive:
        archive[nearby_vector_index] = new_solution
    else:
        original_solution = archive[nearby_vector_index]
        if decomposition(new_solution, nearby_vector, refer_vector, 5) < decomposition(original_solution, nearby_vector, refer_vector, 5):
            archive[nearby_vector_index] = new_solution


def update_archive_random(weight_vectors, archive, new_solution, refer_vector):
    minimum_angle = sys.maxsize
    weight_vectors_size = len(weight_vectors)
    nearby_vector_index = None
    index_pool = [x for x in range(weight_vectors_size)]
    random.shuffle(index_pool)
    threadhold = math.pow(math.pow(3, 0.33) /
                          (2 * math.pi * weight_vectors_size), 0.5)

    # Find the closest vectors
    for i in index_pool[:int(weight_vectors_size / 2)]:
        wv = weight_vectors[i, :]
        cos_value = (np.dot(wv, new_solution)) / \
            math.sqrt((wv.dot(wv.T)) * (new_solution.dot(new_solution.T)))
        angle = math.acos(cos_value)
        if angle < 0.01:
            minimum_angle = angle
            nearby_vector_index = i
            break
        if angle < minimum_angle:
            nearby_vector_index = i
            minimum_angle = angle
    nearby_vector = weight_vectors[nearby_vector_index]

    # Update archive
    if nearby_vector_index not in archive:
        archive[nearby_vector_index] = new_solution
    else:
        original_solution = archive[nearby_vector_index]
        if decomposition(new_solution, nearby_vector, refer_vector, 5) < decomposition(original_solution, nearby_vector, refer_vector, 5):
            archive[nearby_vector_index] = new_solution


def update_EP(EP, new_solution):
    """
    Update new_solution to EP
    """
    for cur_solution in EP:
        if is_dominate(cur_solution, new_solution) is True or np.all(cur_solution == new_solution):
            return None
        else:
            if is_dominate(new_solution, cur_solution) is True:
                remove_from_EP(cur_solution, EP)
    EP.append(new_solution)


def update_BA(BA, new_solution, size_archive):
    """
    Update new_solution to BA
    """
    for cur_solution in BA:
        if is_dominate(cur_solution, new_solution) is True or np.all(cur_solution == new_solution):
            return None
        else:
            if is_dominate(new_solution, cur_solution) is True:
                remove_from_EP(cur_solution, BA)
    if len(BA) < size_archive:
        BA.append(new_solution)
    else:
        if random.random() > 0.5:
            index = random.randint(0, size_archive - 1)
            BA[index] = new_solution


def update_BAHVC(BA, new_solution, size_archive):
    """
    Update new_solution to BA
    """
    for cur_solution in BA:
        if is_dominate(cur_solution, new_solution) is True or np.all(cur_solution == new_solution):
            return None
        else:
            if is_dominate(new_solution, cur_solution) is True:
                remove_from_EP(cur_solution, BA)
    if len(BA) < size_archive:
        BA.append(new_solution)
    else:
        m = len(BA[0])
        hv = pg.hypervolume(BA)
        index = hv.least_contributor([2 for i in range(m)])
        BA.pop(index)
        BA.append(new_solution)


def cal_uniform(archive):
    closest_dis_arr = []
    length_archive = len(archive)
    num = length_archive // 100
    for i in range(length_archive):
        vector_i = archive[i, :]
        min_angle_arr = [sys.maxsize for x in range(num)]
        vector_index_arr = [-1 for x in range(num)]
        for j in range(length_archive):
            vector_j = archive[j, :]
            if np.array_equal(vector_i, vector_j):
                break
            else:
                cos_value = (np.dot(vector_i, vector_j)) / \
                    math.sqrt((vector_i.dot(vector_i.T))
                              * (vector_j.dot(vector_j.T)))
                angle = math.acos(cos_value)
                max_angle = max(min_angle_arr)
                max_index = min_angle_arr.index(max_angle)
                if angle < max_angle:
                    min_angle_arr[max_index] = angle
                    vector_index_arr[max_index] = j
        distance_arr = []
        for v_index in vector_index_arr:
            vector_j = archive[v_index]
            distance = (sum((vector_i - vector_j)**2))**0.5
            distance_arr.append(distance)
        distance_arr = np.array(distance_arr)
        closest_dis_arr.append(np.mean(distance_arr))
    closest_dis_arr = np.array(closest_dis_arr)
    return np.mean(closest_dis_arr), np.var(closest_dis_arr), np.std(closest_dis_arr), np.std(closest_dis_arr) / np.mean(closest_dis_arr)


def remove_from_EP(remove_array, EP):
    for index in range(len(EP)):
        cur_array = EP[index]
        if np.all(cur_array == remove_array):
            del EP[index]
            return True


def generate_init_population(a, b, dimension, size):
    return np.array([[(b - a) * random.random() - abs(a)
                      for j in range(dimension)] for i in range(size)])


def decomposition(fx, coef_vector, refer_vector, theta):
    return decomposition_method.tchebycheff(fx, coef_vector, refer_vector)


def select_result_BHV(archive, remaining_size, refer_point):
    new_archive = list(archive)
    count = len(new_archive)
    while count > remaining_size:
        hv = pg.hypervolume(new_archive)
        index = hv.least_contributor(refer_point)
        new_archive.pop(index)
        count -= 1
    return np.array(new_archive)


def remove_dominated(result):
    n = len(result)
    dominated = [0 for x in range(n)]
    dominated_list = []
    for a in range(n):
        for b in range(n):
            p1 = result[a, :]
            p2 = result[b, :]
            if is_dominate(p1, p2) is True:
                dominated[b] = 1
    for i in range(n):
        if dominated[i] == 0:
            dominated_list.append(result[i, :])
    return np.array(dominated_list)


def is_dominate(x, y):
    """
    Check whether x dominate y(x < y)

    Parameters
    ----------
    x: list or ndarray
    y: list or ndarray

    Returns
    -------
    True for x dominate y
    False for x non-dominate y

    """
    smaller_flag = False
    for i in range(len(x)):
        if(x[i] > y[i]):
            return False
        elif x[i] < y[i]:
            smaller_flag = True
        else:
            pass
    return smaller_flag


def get_weighted_vectors(M, H):
    """Set docstring here.

    Parameters
    ----------
    M: The number of objects
    H: A parameter that influence the number of weight vector

    Returns
    -------
    numpy matrix, every row is a weight vector

    """
    comb = [i for i in range(1, M + H)]
    weight_matrix = []
    comb = list(itertools.combinations(comb, M - 1))
    for space in comb:
        weight = []
        last_s = 0
        for s in space:
            w = (((s - last_s) - 1) / H)
            last_s = s
            weight.append(w)
        weight.append(((M + H - last_s) - 1) / H)
        weight_matrix.append(weight)
    return np.array(weight_matrix)


def mutation(x, rate, upper_bound, lower_bound):
    # Simplely change value
    for i in range(len(x)):
        if random.random() < rate:
            x[i] = lower_bound + random.random() * (upper_bound - lower_bound)
    return x


def crossover(a, b):
    return (a + b) / 2


def imporve(x):
    return x


def random_diff_int(a, b, n):
    """
    Generate n different integer from a to b, [a, b] b included
    n should be bigger than b - a

    Parameters
    ----------
    a: Start from a
    b: End to b
    n: The number of integer

    Returns
    -------
    [n random integer from a to b]

    """
    if n <= 0:
        raise ValueError("n should be positive")
    if a > b:
        t = a
        a = b
        b = t
    if a == b:
        return [a for x in range(n)]
    if n > b - a + 1:
        raise ValueError("n should be bigger than b - a")
    generate_list = [a + x for x in range(b - a + 1)]
    random.shuffle(generate_list)
    return generate_list[:n]


def calculate_n(n, m):
    def ncr(n, r):
        r = min(r, n - r)
        if r == 0:
            return 0
        numer = functools.reduce(operator.mul, range(n, n - r, -1))
        denom = functools.reduce(operator.mul, range(1, r + 1))
        return numer // denom

    h = 1
    cur_n = 0
    while(cur_n < n):
        h += 1
        cur_n = ncr(h + m - 1, m - 1)
    return cur_n, h


def plot_2D_multi(x, m, name):
    theta = math.pi * 2 / m
    x_p = []
    y_p = []
    result = []
    for i in range(m):
        x_p.append(math.sin(i * theta))
        y_p.append(math.cos(i * theta))
    fig = plt.figure(figsize=(9, 9), dpi=200)
    ax = fig.add_subplot(111)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("$y$", fontsize=20)
    ax.scatter(x[:, 0], x[:, 1])
    ax.scatter(x_p[:], y_p[:], c='r')
    plt.savefig("../Testing_result/{0}_node{1}_1.pdf".format(
        name, len(x)), format='pdf', bbox_inches='tight')


def plot_3D_weight_vector(EP, weight_vectors, name):
    # Print 3D graph with EP and weight vectors
    # The enlarge ratio of weight_vectors should change
    enlarge = 1
    fig = plt.figure(figsize=(12, 9), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    plt.title('%s point number: %d weight vector number: %d' %
              (name, len(EP), len(weight_vectors)))
    temp = np.array(EP)
    ax.scatter(temp[:, 0], temp[:, 1], temp[:, 2], c='b')
    ax.view_init(20, 45)


def plot_parallel(polt_data, m, name):
    df = pd.DataFrame(data=polt_data, columns=[i + 1 for i in range(m)])
    df['0'] = pd.Series(1, index=df.index)
    fig = plt.figure(figsize=(12, 9), dpi=200)
    plt.title('%s point number: %d' % (name, len(polt_data)))
    parallel_coordinates(df, class_column='0')
    plt.show()


def plot_3D(EP, name):
    # Print 3D graph of EP
    fig = plt.figure(figsize=(9, 9), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    ax.set_zlim(0, 1.2)
    plt.xlabel("Minimize $f_1(x)$", fontsize=25, labelpad=15)
    plt.ylabel("Minimize $f_2(x)$", fontsize=25, labelpad=15)
    ax.set_zlabel("Minimize $f_3(x)$", fontsize=25, labelpad=15)
    temp = np.array(EP)
    ax.scatter(temp[:, 0], temp[:, 1], temp[:, 2], c='b')
    ax.view_init(20, 45)
    plt.savefig("../Testing_result/{0}_node{1}_1.pdf".format(
        name, len(EP)), format='pdf', bbox_inches='tight')
    plt.close()


def write_file(write_str):
    with open("../Testing_result/Data_result.txt", "a") as f:
        f.write(write_str + "\t\n")


def write_addr(write_str, addr):
    with open(addr, "a") as f:
        f.write(write_str)


def plot_seq(data):
    def decide_shape(i):
        shape = ['-', '*-', '^-', 'o-', 'v-', '<-', '>--', '*--', '^--']
        return shape[i]

    data = np.array(data)
    length = len(data[0]) - 3
    x_axis = [1 / length * i * 100 for i in range(length + 1)]
    fig, ax = plt.subplots(figsize=(18, 9), dpi=200)
    plt.xlim(-2, 100)
    plt.ylim(0, 105)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.xlabel("Percentage of recorded solution (%)", fontsize=25)
    plt.ylabel("Percentage of total solution (%)", fontsize=25)
    for i in range(len(data)):
        data_line = data[i][2:].astype(np.float)
        max_num = max(data_line)
        data_line = (data_line / max_num) * 100
        line_shape = decide_shape(i)
        ax.plot(x_axis, data_line, line_shape, label="{0} dimension:{1}".format(
            data[i][0], data[i][1]), markersize=15, markerfacecolor='none')
        print(data[i][0])
    legend = ax.legend(loc='lower right', shadow=True, fontsize=20)
    plt.savefig("../Testing_result/record.pdf",
                format='pdf', bbox_inches='tight')


def read_seq(addr):
    result = []
    line = []
    with open(addr, "r") as f:
        result = f.readlines()
    result = [x.split(';')[:-1] for x in result]
    return result

