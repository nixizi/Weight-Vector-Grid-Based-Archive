import numpy as np
import math


class Problem_set:

    def __init__(self, problem_name, dimension):
        self._name = problem_name
        self._m = dimension
        self._lower_bound = 0
        self._upper_bound = 1
        self._decision_dimension = 2 * dimension
        if self._name == 'MaF1':
            self._lower_bound = 0
            self._upper_bound = 1
        elif self._name == 'MaF2':
            self._lower_bound = 0
            self._upper_bound = 1
        elif self._name == 'MaF8':
            self._lower_bound = -1
            self._upper_bound = 1
            self._decision_dimension = 2
        elif self._name == 'DTLZ1':
            self._lower_bound = 0
            self._upper_bound = 1
        else:
            pass

    @property
    def name(self):
        return self._name

    def get_boundary(self):
        return self._lower_bound, self._upper_bound

    def get_decision_dimension(self):
        return self._decision_dimension

    def get_result(self, x):
        if self._name == 'MaF1':
            return self.MaF1(x, self._m)
        elif self._name == 'MaF2':
            return self.MaF2(x, self._m)
        elif self._name == 'MaF8':
            return self.MaF8(x, self._m)
        elif self._name == 'DTLZ1':
            return self.DTLZ1(x, self._m)
        else:
            pass

    def DTLZ1(self, x, m):
        d = len(x)
        if d < 3 and m <= d:
            raise ValueError(
                "The length of x should be bigger than 3 or d should less than n")
            return 0
        temp = (x - 0.5) ** 2
        g = sum(temp[m - 1:])
        result = []
        x_multi = 1
        for i in range(m - 1):
            x_multi = x_multi * x[i]
        result.append(x_multi * (1 + g))
        for i in range(m - 1, 1, -1):
            x_multi = 1
            for j in range(i - 1):
                x_multi = x_multi * x[j]
            x_multi = x_multi * (1 - x[i - 1])
            result.append(x_multi * (1 + g))
        result.append((1 - x[0]) * (1 + g))
        return np.array(result)

    def MaF1(self, x, m):
        d = len(x)
        if d < 3 and m <= d:
            raise ValueError(
                "The length of x should be bigger than 3 or d should less than n")
            return 0
        temp = (x - 0.5) ** 2
        g = sum(temp[m - 1:])
        result = []
        x_multi = 1
        for i in range(m - 1):
            x_multi = x_multi * x[i]
        result.append((1 - x_multi) * (1 + g))
        for i in range(m - 1, 1, -1):
            x_multi = 1
            for j in range(i - 1):
                x_multi = x_multi * x[j]
            x_multi = x_multi * (1 - x[i - 1])
            result.append((1 - x_multi) * (1 + g))
        result.append(x[0] * (1 + g))
        return np.array(result)

    def MaF2(self, x, m):
        d = len(x)
        k = d - m + 1
        theta = math.pi / 2 * (x / 2 + 1 / 4)
        g = []
        for i in range(1, m, 1):
            # [1, 2, 3, ..., m - 1]
            temp = 0
            from_index = m + (i - 1) * math.floor(k / m)
            to_index = m + i * math.floor(k / m) - 1
            for j in range(from_index, to_index + 1, 1):
                temp += math.pow(((x[j - 1] / 2 + 1 / 4) - 0.5), 2)
            g.append(temp)
        temp = 0
        for i in range(m + (m - 1) * math.floor(k / m), d + 1, 1):
            temp += math.pow(((x[i - 1] / 2 + 1 / 4) - 0.5), 2)
        g.append(temp)
        result = []
        temp = 1
        for i in range(1, m + 1, 1):
            temp *= math.cos(theta[i - 1])
        result.append(temp * (1 + g[0]))
        for i in range(m - 1, 1, -1):
            temp = 1
            for j in range(1, i, 1):
                temp *= math.cos(theta[j - 1])
            temp *= math.sin(theta[i - 1]) * (1 + g[m - i])
            result.append(temp)
        result.append(math.sin(theta[0]) * (1 + g[m - 1]))
        return np.array(result)

    def MaF8(self, x, m):
        if len(x) != 2:
            raise ValueError("The length of x should be 2")
            return 0
        theta = math.pi * 2 / m
        x_p = []
        y_p = []
        result = []
        for i in range(m):
            x_p.append(math.sin(i * theta))
            y_p.append(math.cos(i * theta))
        for i in range(len(x_p)):
            result.append(math.sqrt((x[0] - x_p[i])
                                    ** 2 + (x[1] - y_p[i]) ** 2))
        return np.array(result)
