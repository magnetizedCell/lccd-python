import numpy as np

def syntesize_data(X, Y, T, num_points):
    xs = np.random.randint(0, X, num_points)
    ys = np.random.randint(0, Y, num_points)
    heights = np.random.randint(3, 10, num_points)
    widths = np.random.randint(3, 10, num_points)
    intensities = np.random.uniform(0.7, 1.0, num_points)

    arr = np.zeros((X, Y), dtype=np.float64)
    for i in range(num_points):
        x, y = xs[i], ys[i]
        height, width = heights[i], widths[i]
        intensity = intensities[i]
        arr[y: min(Y, y+height), x: min(X, x+width)] = intensity
    
    stacked_arr = np.zeros((X, Y, T))
    x_roll, y_roll = 0, 0
    for frame in range(T):
        x_roll = np.random.randint(-2, 2)
        y_roll = np.random.randint(-2, 2)
        tmp = np.roll(arr, (x_roll, y_roll)) * np.random.normal(1, 0.1, size=(X, Y)) + np.random.uniform(0, 0.2, size=(X, Y))
        stacked_arr[:, :, frame] = tmp
    return stacked_arr
