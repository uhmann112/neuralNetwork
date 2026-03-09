import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return -(0.0001 * x**3) + 50


def generate_data(n=200):
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    # Labels: 1 = über der Funktion, 0 = unter der Funktion
    labels = (y > f(x)).astype(int)
    # Trainingsmatrix
    X = np.column_stack((x, y))
    return X, labels


# ---------------------------------------------------------
# 3) Plotten der Funktion + Punkte + Bereiche
# ---------------------------------------------------------
def plot_data(X, labels):
    fig, ax = plt.subplots()

    # Funktion plotten
    xs = np.linspace(0, 100, 500)
    ax.plot(xs, f(xs), color="black", linewidth=2)

    # Punkte einfärben
    colors = np.where(labels == 1, "red", "blue")
    ax.scatter(X[:, 0], X[:, 1], c=colors)

    # Bereiche einfärben
    ax.fill_between(xs, f(xs), 0, color="blue", alpha=0.15)
    ax.fill_between(xs, f(xs), 100, color="red", alpha=0.15)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Trainingsdaten + Funktion")

    plt.show()


# ---------------------------------------------------------
# 4) Ausführen
# ---------------------------------------------------------
X, labels = generate_data(n=200)
plot_data(X, labels)

