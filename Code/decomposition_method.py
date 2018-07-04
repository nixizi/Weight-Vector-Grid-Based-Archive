import sys
import numpy as np


def weighted_sum(fx, coef_vector):
    """
    Weighted Sum approach
        :param fx: numpy array. F(x)
        :param coef_vector: nummpy array. The coefficient vector. Lambda
    """
    if len(fx) != len(coef_vector):
        raise ValueError(
            "Length of fx is not same as length of coefficient vector")
    else:
        return np.dot(fx, coef_vector)


def tchebycheff(fx, coef_vector, refer_vector):
    """
    Tchebycheff approach
        :param fx: numpy array. F(x)
        :param coef_vector: numpy array. The coefficient vector. Lambda
        :param refer_point: numpy array. The refference point. Z*
    """
    if not(len(fx) == len(coef_vector) == len(refer_vector)):
        raise ValueError(
            "Length of fx is not same as length of coef_vector or coefficient vector")
    else:
        maximum_distance = (coef_vector * abs(fx - refer_vector)).max(axis=0)
    return maximum_distance


def boundary_intersection_max(fx, coef_vector, refer_vector, theta):
    """
    Boundary intersection approach
        :param fx: numpy array. F(x)
        :param coef_vector: numpy array. The coefficient vector. Lambda
        :param refer_vector: numpy array. The refference point. z*
        :param theta: theta parameter
    """
    d1 = (refer_vector - fx).dot(coef_vector) / \
        np.sqrt(coef_vector.dot(coef_vector))
    d2 = fx - (refer_vector - d1 * coef_vector)
    d2 = np.sqrt(d2.dot(d2))
    return d1 + theta * d2


def boundary_intersection_min(fx, coef_vector, refer_vector, theta):
    """
    Boundary intersection approach
        :param fx: numpy array. F(x)
        :param coef_vector: numpy array. The coefficient vector. Lambda
        :param refer_vector: numpy array. The refference point. z*
        :param theta: theta parameter
    """
    d1 = (fx - refer_vector).dot(coef_vector) / \
        np.sqrt(coef_vector.dot(coef_vector))
    d2 = fx - (refer_vector + d1 * coef_vector)
    d2 = np.sqrt(d2.dot(d2))
    return d1 + theta * d2
