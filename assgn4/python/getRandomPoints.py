import numpy as np


def get_random_points(I, alpha):

    # -----fill in your implementation here --------
    x_coord = np.random.randint(0, I.shape[1], alpha)
    y_coord = np.random.randint(0, I.shape[0], alpha)
    points = np.transpose(np.array([x_coord, y_coord]))

    # ----------------------------------------------

    return points
