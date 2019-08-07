def grad_desc_approx(fcn, guess=(0, 0, 0),
                     eps=10**-3, alpha=10**-3, tol=10**-8):
    """This function simulates 3-d gradient descent by using
    finite differences instead of derivatives"""
    import numpy as np
    X0 = guess[0]
    Y0 = guess[1]
    Z0 = guess[2]
    ddx = (fcn(X0+eps, Y0, Z0) - fcn(X0, Y0, Z0)) / eps
    ddy = (fcn(X0, Y0+eps, Z0) - fcn(X0, Y0, Z0)) / eps
    ddz = (fcn(X0, Y0, Z0+eps) - fcn(X0, Y0, Z0)) / eps
    step = alpha * np.array([ddx, ddy, ddz])
    new_coords = np.array([X0, Y0, Z0]) - step
    X1, Y1, Z1 = new_coords[0], new_coords[1], new_coords[2]
    diff = abs(fcn(X1, Y1, Z1) - fcn(X0, Y0, Z0))
    ctr = 1
    while diff > tol:
        X0, Y0, Z0 = X1, Y1, Z1
        ddx = (fcn(X0+eps, Y0, Z0) - fcn(X0, Y0, Z0)) / eps
        ddy = (fcn(X0, Y0+eps, Z0) - fcn(X0, Y0, Z0)) / eps
        ddz = (fcn(X0, Y0, Z0+eps) - fcn(X0, Y0, Z0)) / eps
        step = alpha * np.array([ddx, ddy, ddz])
        new_coords = np.array([X0, Y0, Z0]) - step
        X1, Y1, Z1 = new_coords[0], new_coords[1], new_coords[2]
        diff = abs(fcn(X1, Y1, Z1) - fcn(X0, Y0, Z0))
        ctr += 1
        if ctr > 10**6:
            break

    return np.array([X1, Y1, Z1])