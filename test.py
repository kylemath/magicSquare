import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

def generate_magic_square(n):
    if n % 2 == 1:  # Odd order magic square
        magic_square = np.zeros((n, n), dtype=int)
        num = 1
        i, j = 0, n // 2

        while num <= n**2:
            magic_square[i, j] = num
            num += 1
            newi, newj = (i - 1) % n, (j + 1) % n
            if magic_square[newi, newj]:
                i += 1
            else:
                i, j = newi, newj
    elif n % 4 == 0:  # Doubly even order magic square
        magic_square = np.zeros((n, n), dtype=int)
        num = 1
        for i in range(n):
            for j in range(n):
                magic_square[i, j] = num
                num += 1

        # Swap the values in the magic square
        for i in range(n):
            for j in range(n):
                if (i % 4 == j % 4) or ((i + j) % 4 == 3):
                    magic_square[i, j] = n * n + 1 - magic_square[i, j]
    elif n % 2 == 0:  # Singly even order magic square
        magic_square = np.zeros((n, n), dtype=int)
        num = 1
        half_n = n // 2
        sub_square = generate_magic_square(half_n)

        # Fill the quadrants
        magic_square[:half_n, :half_n] = sub_square
        magic_square[:half_n, half_n:] = sub_square + half_n * half_n
        magic_square[half_n:, :half_n] = sub_square + 2 * half_n * half_n
        magic_square[half_n:, half_n:] = sub_square + 3 * half_n * half_n

        # Swap the values in the magic square
        for i in range(half_n):
            for j in range(half_n // 2):
                if j != half_n // 2 or i != 0:
                    magic_square[i, j], magic_square[half_n + i, j] = magic_square[half_n + i, j], magic_square[i, j]

    else:
        raise ValueError("Magic square generation is only implemented for odd integers and even integers greater than 4.")

    return magic_square

def visualize_magic_square(magic_square):
    n = magic_square.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    x, y = np.meshgrid(x, y)
    z = magic_square * 2  # Double the vertical scaling

    # Prepare points for the convex hull
    points = np.array([(x[i, j], y[i, j], z[i, j] + 1) for i in range(n) for j in range(n)])  # Raise slightly above

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    # Adding spheres at each point
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=10)  # Scatter spheres

    # Create and plot the convex hull
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], color='blue', alpha=0.5)

    # Optionally, plot the vertices of the convex hull
    ax.scatter(points[hull.vertices, 0], points[hull.vertices, 1], points[hull.vertices, 2], color='green', s=20)  # Hull vertices

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Value')
    ax.set_title('Magic Square Points with Convex Hull (Vertical Scaling Doubled)')

    plt.show()

n = 62  # Size of the magic square (can be changed to 8, 10, etc.)
magic_square = generate_magic_square(n)
print(magic_square)
visualize_magic_square(magic_square)