import math
import numpy as np
import copy
import random
import decomposition_method
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import pygmo as pg
import time
import pandas as pd
from pandas.plotting import parallel_coordinates
import problem_set
import operator
import functools
import support_method

count = 0
count_total = 0


def MOEA_D_archive(prob_name, size_main, size_archive, num_of_object, ratio_of_neighbor, total_evaluated, model):
    start_time = time.time()
    # The number of object function
    m = num_of_object

    prob = problem_set.Problem_set(prob_name, m)
    lower_bound, upper_bound = prob.get_boundary()

    remain_size = 200
    if prob_name == 'MaF1':
        remain_size = 270
    elif prob_name == 'MaF2':
        remain_size = 210
    else:
        remain_size = 860

    # The size of population in small grid weight vector and in large grid weight vector
    n_main = size_main
    n_archive = size_archive

    # Dimension of decision space
    d_decision = prob.get_decision_dimension()

    # The number of evaluated individuals number
    num_of_evaluation = 0

    # Current object value of weight_vector
    FV = []

    # Store Population
    archive = {}

    # Store Decision Solution
    # solution_dict = {}

    n_main, h_in_main = support_method.calculate_n(n_main, m)
    n_archive, h_in_archive = support_method.calculate_n(n_archive, m)

    # The number of closest weight vector to weight vector i
    num_of_closest_vector = math.ceil(n_main * ratio_of_neighbor)

    refer_vector = []
    weight_vectors = support_method.get_weighted_vectors(m, h_in_main)
    archive_weight_vectors = support_method.get_weighted_vectors(
        m, h_in_archive)

    # Init population
    population = support_method.generate_init_population(
        lower_bound, upper_bound, d_decision, n_main)

    # Calculate the closest weight vector to weight vector i
    closest_weight_vectors = []
    for i in range(n_main):
        temp_close = []
        vector_A = weight_vectors[i, :]
        for j in range(n_main):
            vector_B = weight_vectors[j, :]
            if vector_A is vector_B:
                continue
            euclidean_distance = sum((vector_A - vector_B)**2)
            temp_close.append((j, euclidean_distance))
        temp_close.sort(key=lambda x: x[1])
        temp_close = [temp_close[x][0] for x in range(num_of_closest_vector)]
        closest_weight_vectors.append(temp_close)
    closest_weight_vectors = np.array(closest_weight_vectors, dtype=int)

    # Initialize FV
    for i in range(n_main):
        solution = prob.get_result(population[i])
        FV.append(solution)
    FV = np.array(FV)

    # Initialize reference vector z*
    refer_vector = np.array(FV).min(axis=0)
    refer_vector = np.array(refer_vector)

    for solution in FV:
        support_method.update_archive(
            archive_weight_vectors, archive, solution, refer_vector, update_type=model)

    name = "MOEAD_{5}_{0}_m{1}_S{2}_L{3}_E{4}".format(
        prob.name, m, n_main, n_archive, total_evaluated, model)

    generation = 0
    stop_flag = False
    while stop_flag is False:
        generation += 1
        for i in range(n_main):
            # Randomly select two index k, l from the set of neightbor
            [k, l] = support_method.random_diff_int(
                0, num_of_closest_vector - 1, 2)
            k = closest_weight_vectors[i, k]
            l = closest_weight_vectors[i, l]

            # Generate new solution base on genetic operator
            child = support_method.crossover(population[k], population[l])
            child = support_method.mutation(
                child, 0.3, upper_bound, lower_bound)

            # Improve by problem specific method
            child = support_method.imporve(child)

            # Calculate the object result
            new_solution = prob.get_result(child)

            # Update reference point z*
            for j in range(m):
                if new_solution[j] < refer_vector[j]:
                    refer_vector[j] = new_solution[j]
            new_solution = np.array(new_solution)

            for index in closest_weight_vectors[i, :]:
                if(support_method.decomposition(new_solution, weight_vectors[index, :], refer_vector, 0) < support_method.decomposition(FV[index, :], weight_vectors[index, :], refer_vector, 0)):
                    population[index] = copy.deepcopy(np.array(child))
                    FV[index] = copy.deepcopy(new_solution)

            # Update new solution to archive
            support_method.update_archive(
                archive_weight_vectors, archive, new_solution, refer_vector, update_type=model)
            num_of_evaluation += 1
            # When the number of evaluated individuls meet the stop criteria
            if num_of_evaluation == total_evaluated:
                stop_flag = True
                break

    end_time_main = time.time()
    best_solutions = []
    for key in archive:
        best_solutions.append(archive[key])
    best_solutions = np.array(best_solutions)
    best_solutions = support_method.remove_dominated(best_solutions)
    hv_refer = [1.1 for i in range(m)]
    new_archive = []
    for i in range(len(best_solutions)):
        point = best_solutions[i]
        if max(point) < max(hv_refer):
            new_archive.append(point)
    new_archive = support_method.select_result_BHV(
        new_archive, remain_size, hv_refer)
    print("Length of archive is %d" % len(new_archive))
    hv = pg.hypervolume(new_archive)
    hv_result = hv.compute(hv_refer)
    return hv_result, end_time_main - start_time


def MOEA_D_archive_generation(prob_name, size_main, size_archive, num_of_object, ratio_of_neighbor, total_evaluated, ratio_of_record):
    start_time = time.time()
    model = 'archive'
    # The number of object function
    m = num_of_object

    prob = problem_set.Problem_set(prob_name, m)
    lower_bound, upper_bound = prob.get_boundary()

    # The size of population in small grid weight vector and in large grid weight vector
    n_main = size_main
    n_archive = size_archive

    # Dimension of decision space
    d_decision = prob.get_decision_dimension()

    # The number of evaluated individuals number
    num_of_evaluation = 0

    # Current object value of weight_vector
    FV = []

    # Store Population
    archive = {}

    # Store Decision Solution
    solution_dict = {}

    n_main, h_in_main = support_method.calculate_n(n_main, m)
    n_archive, h_in_archive = support_method.calculate_n(n_archive, m)

    # The number of closest weight vector to weight vector i
    num_of_closest_vector = math.ceil(n_main * ratio_of_neighbor)

    refer_vector = []
    weight_vectors = support_method.get_weighted_vectors(m, h_in_main)
    archive_weight_vectors = support_method.get_weighted_vectors(
        m, h_in_archive)

    # Init population
    population = support_method.generate_init_population(
        lower_bound, upper_bound, d_decision, n_main)

    num_of_evaluation = n_main

    # Calculate the closest weight vector to weight vector i
    closest_weight_vectors = []
    for i in range(n_main):
        temp_close = []
        vector_A = weight_vectors[i, :]
        for j in range(n_main):
            vector_B = weight_vectors[j, :]
            if vector_A is vector_B:
                continue
            euclidean_distance = sum((vector_A - vector_B)**2)
            temp_close.append((j, euclidean_distance))
        temp_close.sort(key=lambda x: x[1])
        temp_close = [temp_close[x][0] for x in range(num_of_closest_vector)]
        closest_weight_vectors.append(temp_close)
    closest_weight_vectors = np.array(closest_weight_vectors, dtype=int)

    # Initialize FV
    for i in range(n_main):
        solution = prob.get_result(population[i])
        FV.append(solution)
        solution_dict[tuple(solution)] = copy.deepcopy(population[i])
    FV = np.array(FV)

    # Initialize reference vector z*
    refer_vector = np.array(FV).min(axis=0)
    refer_vector = np.array(refer_vector)

    name = "MOEAD_archive_generation_{0}_m{1}_S{2}_L{3}_E{4}".format(
        prob.name, m, n_main, n_archive, total_evaluated)

    generation = 0
    stop_flag = False
    while stop_flag is False:
        generation += 1
        for i in range(n_main):
            # Randomly select two index k, l from the set of neightbor
            [k, l] = support_method.random_diff_int(
                0, num_of_closest_vector - 1, 2)
            k = closest_weight_vectors[i, k]
            l = closest_weight_vectors[i, l]

            # Generate new solution base on genetic operator
            child = support_method.crossover(population[k], population[l])
            child = support_method.mutation(
                child, 0.3, upper_bound, lower_bound)

            # Improve by problem specific method
            child = support_method.imporve(child)

            # Calculate the object result
            new_solution = prob.get_result(child)

            solution_dict[tuple(new_solution)] = copy.deepcopy(child)

            # Update reference point z*
            for j in range(m):
                if new_solution[j] < refer_vector[j]:
                    refer_vector[j] = new_solution[j]
            new_solution = np.array(new_solution)

            for index in closest_weight_vectors[i, :]:
                if(support_method.decomposition(new_solution, weight_vectors[index, :], refer_vector, 0) < support_method.decomposition(FV[index, :], weight_vectors[index, :], refer_vector, 0)):
                    population[index] = copy.deepcopy(np.array(child))
                    FV[index] = copy.deepcopy(new_solution)

            num_of_evaluation += 1
            if num_of_evaluation > total_evaluated * (1 - ratio_of_record):
                # Update new solution to archive
                support_method.update_archive(
                    archive_weight_vectors, archive, new_solution, refer_vector, update_type=model)
            # When the number of evaluated individuls meet the stop criteria
            if num_of_evaluation == total_evaluated:
                stop_flag = True
                break

    for point in FV:
        support_method.update_archive_origin(
            archive_weight_vectors, archive, point, refer_vector)
    end_time_main = time.time()
    best_solutions = []
    for key in archive:
        best_solutions.append(archive[key])
    best_solutions = np.array(best_solutions)
    decision_result = []
    for key in best_solutions:
        decision_result.append(solution_dict[tuple(key)])
    hv_refer = [1.1 for i in range(m)]
    hv = pg.hypervolume(best_solutions)
    hv_result = hv.compute(hv_refer)
    return hv_result, end_time_main - start_time
