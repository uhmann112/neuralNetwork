import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import main
import cProfile


x =[]
y=[]
plt.ion()

def f(x):
    return -(0.0001 * x**3) + 50


def generate_data(n):
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    lab = np.array([1 if f(xi) <= yi else 0 for xi, yi in zip(x, y)])
    output=[]
    for i in range(n):
        storage=[]
        output.append([x[i],y[i],lab[i]])

    return output , x, y, lab

def plotTestPoints(x,y):
    plt.scatter(x, y, c=lab, cmap="coolwarm")
    plt.plot(np.linspace(0,100,200), f(np.linspace(0,100,200)), color="black")
    plt.title("Generated Data + f(x)")

def make_grid(xmin=0, xmax=100, ymin=0, ymax=100, step=1):
    xs = np.arange(xmin, xmax, step)
    ys = np.arange(ymin, ymax, step)

    X, Y = np.meshgrid(xs, ys)

    # zu einer Liste von Punkten machen
    grid = np.column_stack((X.ravel(), Y.ravel()))
    return grid

grid = make_grid(step=1)





data, x, y, lab = generate_data(20)
plotTestPoints(x,y)
epochs=20

for i in range(epochs):
    
    prediction = np.array([main.predict(x) for x in grid])

    # Wahrscheinlichkeit für Klasse 1
    p = prediction[:, 1]

    # Punkte nahe 0.5 behalten
    tolerance = 0.05
    mask = np.abs(p - 0.5) < tolerance

    # WICHTIG: die GRID-PUNKTE filtern, nicht die Predictions
    boundary_points = grid[grid]

    # Nach x sortieren
    boundary_points = boundary_points[boundary_points[:, 0].argsort()]

    # Linie plotten
    plt.plot(boundary_points[:, 0], boundary_points[:, 1], color="red")

    plt.pause(0.001)
    main.runNetwork(data)

#cProfile.run("main.runNetwork(data)")


#[Finished in 90.7s]









