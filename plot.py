import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import main



plt.ion()


def f(x):
    return -(0.0001 * x**3) + 50

def generate_data(n):
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    lab = np.array([1 if f(xi) <= yi else 0 for xi, yi in zip(x, y)])
    output = [[x[i], y[i], lab[i]] for i in range(n)]
    return output, x, y, lab

def plotTestPoints(x, y, lab):
    plt.scatter(x, y, c=lab, cmap="coolwarm")
    plt.plot(np.linspace(0, 100, 200), f(np.linspace(0, 100, 200)), color="blue",linewidth=3)
    plt.title("Generated Data + f(x)")

def make_grid(xmin=0, xmax=100, ymin=0, ymax=100, step=3):
    xs = np.arange(xmin, xmax, step)
    ys = np.arange(ymin, ymax, step)
    X, Y = np.meshgrid(xs, ys)
    grid = np.column_stack((X.ravel(), Y.ravel()))
    return grid

grid = make_grid(step=1)
unique_xs = np.unique(grid[:, 0])  # ← FIX 1: echte X-Werte aus dem Grid

data, x, y, lab = generate_data(500)
plotTestPoints(x, y, lab)  # ← FIX 2: lab übergeben (war vorher nicht übergeben)

epochs = 6
boundary_line = None  # ← FIX 3: Handle für die Linie merken

for i in range(epochs):
    prediction = np.array([main.predict(p) for p in grid])
    p = prediction[:, 1]

    boundary_x = []
    boundary_y = []

    for xi in unique_xs:  # ← FIX 1: über echte Grid-X-Werte iterieren
        mask = grid[:, 0] == xi
        ys_for_x = grid[mask][:, 1]
        p_for_x = p[mask]
        idx = np.argmin(np.abs(p_for_x - 0.5))
        boundary_x.append(xi)
        boundary_y.append(ys_for_x[idx])


    if boundary_line is not None:
        boundary_line[0].remove()  # ← FIX 3: alte Linie löschen

    boundary_line = plt.plot(boundary_x, boundary_y, color="red", linewidth=2)
    plt.fill_between(boundary_x, boundary_y, 100,color="#f472b6",  alpha=0.65)  # oben
    plt.fill_between(boundary_x, boundary_y, 0,   color="#818cf8", alpha=0.65)  # unten
    plt.title(f"Epoch {i+1}/{epochs}")
    plt.pause(0.1)

    main.runNetwork(data)

plt.ioff()
plt.show()