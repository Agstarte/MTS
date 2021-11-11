import os

import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_rel

clfs = list()

metrics = ['minkowski', 'euclidean']
k_values = [3, 5, 7]

for k in k_values:
    for metric in metrics:
        clfs.append(f'K{k}_{metric}')

for result in os.listdir('results'):

    print('\n\n' + f'='*110 +
          f'\n{"".join([s for s in result if s.isdigit()])} features:\n')

    scores = np.load(os.path.join('results', result))

    alfa = .05
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])

    headers = clfs
    names_column = np.array([[clf] for clf in clfs])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\nAdvantage:\n", advantage_table)

    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("\nStatistically significantly better:\n", stat_better_table)