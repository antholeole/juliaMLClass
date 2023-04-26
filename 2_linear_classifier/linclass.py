import numpy as np
import matplotlib.pyplot as plt

import numpy # import again
import matplotlib.pyplot # import again

import numpy.linalg
import numpy.random


def generate_data(Para1, Para2, seed=0):
    """Generate binary random data

    Para1, Para2: dict, {str:float} for each class,
      keys are mx (center on x axis), my (center on y axis),
               ux (sigma on x axis), ux (sigma on y axis),
               y (label for this class)
    seed: int, seed for NUMPy's random number generator. Not Python's random.

    """
    numpy.random.seed(seed)
    X1 = numpy.vstack((numpy.random.normal(Para1['mx'], Para1['ux'], Para1['N']),
                       numpy.random.normal(Para1['my'], Para1['uy'], Para1['N'])))
    X2 = numpy.vstack((numpy.random.normal(Para2['mx'], Para2['ux'], Para2['N']),
                       numpy.random.normal(Para2['my'], Para2['uy'], Para2['N'])))
    Y = numpy.hstack(( Para1['y']*numpy.ones(Para1['N']),
                       Para2['y']*numpy.ones(Para2['N'])))
    X = numpy.hstack((X1, X2))
    X = numpy.transpose(X)
    print(f"{X=} {Y=}")
    return X, Y

def plot_data_hyperplane(X, y, w, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented
    y: 1-D numpy array, the labels
    w: 1-by-3 numpy array, the last element of which is the bias term

    Examples
    --------------
    >>> X = numpy.array([[1,2], \
                         [4,5], \
                         [7,8]])
    >>> y = numpy.array([1,-1,1])
    >>> w = [1, 2, -10]
    >>> filename = "test.png"
    >>> plot_data_hyperplane(X, y, w, filename)
    """

    # your code below
    plt.xlim(numpy.min(X[:,0]), numpy.max(X[:,0]))
    plt.ylim(numpy.min(X[:,1]), numpy.max(X[:,1]))

    X1 = X[y == +1]
    X2 = X[y == -1]

    plt.plot(X1[:, 0], X1[:, 1], 'ro')
    plt.plot(X2[:, 0], X2[:, 1], 'bo')

    x_ticks = numpy.array([numpy.min(X[:,0]), numpy.max(X[:,0])])
    y_ticks = -1 * (x_ticks * w[0] + w[2]) / w[1]

    plt.plot(x_ticks, y_ticks, 'k-')

    plt.savefig(filename)
    plt.close('all')


    # your code above

def learn_and_visual_mse(X, y, filename):
    w = np.array([0,0,0]) # just a placeholder

    X_aug = np.hstack((X, np.ones((X.shape[0], 1))))
    X_transpose = np.transpose(X_aug)
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_transpose, X_aug)), X_transpose), y)

    plot_data_hyperplane(X, y, w, filename)

    return w

def learn_and_visual_fisher(X, y, filename):
    w = np.array([0,0,0])

    c1, c2 = X[y == 1], X[y == -1]
    m1 = (len(c1) ** -1) * np.sum()

    plot_data_hyperplane(X, y, w, filename)
    return w


if __name__ == "__main__":
    import doctest
    doctest.testmod(globs={
        'learn_and_visual_mse': True
    })