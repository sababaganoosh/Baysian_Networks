def inference(z, py, pz, q, log):
    """
    Variable elimination program:
        z: a sequence of observed values
        i: index of the circuit input to infer
        p: noisy parameter
        q: prior probability
    """

    def X(x, q=q):
        # return input probability
        if log == False:
            return q if x == 1 else 1 - q
        else:
            return np.log(q) if x == 1 else np.log(1 - q)

    def nxor(out, in1, in2, gate='u', py=py, pz=pz):
        # 'u' for upper gate using py and 'l' for lower gate using pz
        p = py if gate == 'u' else pz
        # return noisy XOR probabilities
        if in1 == 1 or in2 == 1:
            if log == False:
                return 1 - p if out == 1 else p
            else:
                return np.log(1 - p) if out == 1 else np.log(p)
        else:
            if log == False:
                return p if out == 1 else 1 - p
            else:
                return np.log(p) if out == 1 else np.log(1 - p)

    def my1():
        # create max_y1 table with associated argmax values in dictionary
        argmax_dict['Y%s' % 1] = {}

        z1 = int(z[0])

        for x2 in [0, 1]:
            for y2 in [0, 1]:
                argmax = np.array([])
                for y1 in [0, 1]:
                    # table created using probabilities of z1 and mx1
                    if log == False:
                        prob = nxor(z1, y1, y2, 'l') * argmax_dict['X1'][y1, x2][1]
                    else:
                        prob = nxor(z1, y1, y2, 'l') + argmax_dict['X1'][y1, x2][1]

                    argmax = np.append(argmax, prob)

                argmax_dict['Y%s' % 1][y2, x2] = [np.argmax(argmax), np.max(argmax)]

        i = 1
        result = mxi_right(i)
        return result

    def mxi_right(i):
        # create max_xi table with associated argmax values in dictionary
        i += 1
        argmax_dict['X%s' % i] = {}

        if i == L + 2:
            argmax = np.array([])
            for xL2 in [0, 1]:
                # max prob determined using probabilities of xL2 and myL1
                if log == False:
                    prob = X(xL2) * argmax_dict['Y%s' % (i - 1)][xL2][1]
                else:
                    prob = X(xL2) + argmax_dict['Y%s' % (i - 1)][xL2][1]

                argmax = np.append(argmax, prob)

                argmax_dict['X%s' % i][xL2] = [np.argmax(argmax), np.max(argmax)]

            result = argmax_dict
        elif i == 2:
            for yi in [0, 1]:
                for xii in [0, 1]:
                    argmax = np.array([])
                    for xi in [0, 1]:
                        # table created using probabilities of x2, y2, and my1
                        if log == False:
                            prob = X(xi) * nxor(yi, xi, xii, 'u') * argmax_dict['Y%s' % (i - 1)][yi, xi][1]
                        else:
                            prob = X(xi) + nxor(yi, xi, xii, 'u') + argmax_dict['Y%s' % (i - 1)][yi, xi][1]

                        argmax = np.append(argmax, prob)

                    argmax_dict['X%s' % i][yi, xii] = [np.argmax(argmax), np.max(argmax)]

            result = my2(i)
        else:
            for yi in [0, 1]:
                for xii in [0, 1]:
                    argmax = np.array([])
                    for xi in [0, 1]:
                        # table created using probabilities of xi, yi, and m*y_i-1
                        if log == False:
                            prob = X(xi) * nxor(yi, xi, xii, 'u') * argmax_dict['Y%s' % (i - 1)][xi][1]
                        else:
                            prob = X(xi) + nxor(yi, xi, xii, 'u') + argmax_dict['Y%s' % (i - 1)][xi][1]

                        argmax = np.append(argmax, prob)

                    argmax_dict['X%s' % i][yi, xii] = [np.argmax(argmax), np.max(argmax)]

            result = myi_right(i)

        return result

    def my2(i):
        # create max_y2 table with associated argmax values in dictionary
        argmax_dict['Y%s' % 2] = {}

        zi1 = int(z[i - 2])
        zi2 = int(z[i - 3])
        for x3 in [0, 1]:
            argmax = np.array([])
            for y2 in [0, 1]:
                # table created using probabilities of m*x2
                prob = argmax_dict['X%s' % (i)][y2, x3][1]

                argmax = np.append(argmax, prob)

            argmax_dict['Y%s' % i][x3] = [np.argmax(argmax), np.max(argmax)]

        result = mxi_right(i)
        return result

    def myi_right(i):
        # create max_yi table with associated argmax values in dictionary
        argmax_dict['Y%s' % i] = {}
        zi1 = int(z[i - 2])
        zi2 = int(z[i - 3])
        for xii in [0, 1]:
            argmax = np.array([])
            for yi in [0, 1]:
                # table created using probabilities of z_i-1 and m*xi
                if log == False:
                    prob = nxor(zi1, yi, zi2, 'l') * argmax_dict['X%s' % (i)][yi, xii][1]
                else:
                    prob = nxor(zi1, yi, zi2, 'l') + argmax_dict['X%s' % (i)][yi, xii][1]

                argmax = np.append(argmax, prob)

            argmax_dict['Y%s' % i][xii] = [np.argmax(argmax), np.max(argmax)]

        result = mxi_right(i)
        return result

    import numpy as np

    # infer from Zs
    L = len(z)

    # creeate dictionary to store max and argmax tables
    argmax_dict = {}
    argmax_dict['X%s' % 1] = {}
    # create max_x1 table with associated argmax values in dictionary
    for y_1 in [0, 1]:
        for x_2 in [0, 1]:
            argmax = np.array([])
            for x_1 in [0, 1]:
                # table created using probabilities of x1 and y1
                if log == False:
                    prob = X(x_1) * nxor(y_1, x_1, x_2, 'u')
                else:
                    prob = X(x_1) + nxor(y_1, x_1, x_2, 'u')

                argmax = np.append(argmax, prob)

            argmax_dict['X%s' % 1][y_1, x_2] = [np.argmax(argmax), np.max(argmax)]

    return my1()





