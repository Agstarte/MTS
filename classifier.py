import numpy as np

from sklearn.feature_selection import r_regression
from sklearn.neighbors import KNeighborsClassifier


dataset = np.genfromtxt("wisconsin.csv", delimiter=",")
X = dataset[1:, :-1]
y = dataset[1:, -1].astype(int)

ranking = r_regression(X, y, center=True)


# knn – dla 3 różnych wartości k oraz dla 2 różnych miar odległości
# (w tym euklidesowej).
metrics = ['minkowski', 'euclidean']
k_values = [3, 5, 7]

for k in k_values:
    for metric in metrics:
        pass

# Ewaluacja z wykorzystaniem protokołu badawczego 5 razy powtórzonej
# 2-krotnej walidacji krzyżowej (ang. Cross-validation). Jakość klasyfikacji
# (poprawność diagnozy) należy mierzyć metryką dokładności (accuracy).

# Badania należy przeprowadzić dla różnej liczby cech (poczynając od jednej
# - najlepszej wg. wyznaczonego rankingu, a następnie dokładać kolejno po jednej
# (również według wyznaczonego rankingu) tak długo, aż
# zostanie znaleziona najlepsza liczba cech. Jeżeli cech jest mało (< 7), to
# przeprowadzić badania dla wszystkich cech.


# Dla każdego pojedynczego eksperymentu (pojedynczy eksperyment to
# doświadczalne wyznaczenie jakości klasyfikacji dla danego algorytmu,
# danych wartości parametrów algorytmu i dla danej liczby cech) należy
# przedstawić wyniki w formie uśrednionej (względem 5 powtórzeń metody
# 2-krotnej walidacji krzyżowej).

# – Należy przeprowadzić analizę statystyczną (np. korzystając z testu
# parowego t-studenta) porównującą ze sobą wyniki uzyskane przez każdy z
# sześciu klasyfikatorów osobno dla każdej badanej liczby cech.
