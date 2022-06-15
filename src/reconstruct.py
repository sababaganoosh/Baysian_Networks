def reconstruct(z, py, pz, q, argmax_dict, log):
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

    def mxi_left(i, yi, xii, argmax_dict):
        if i == 2:
            x2 = argmax_dict['X2'][yi, xii][0]
            assignment_dict['X%s' % i] = x2
            if log == False:
                result = nxor(yi, x2, xii, 'u') * X(x2) * my1(x2, yi, argmax_dict)
            else:
                result = nxor(yi, x2, xii, 'u') + X(x2) + my1(x2, yi, argmax_dict)
        elif i == 1:
            x1 = argmax_dict['X1'][yi, xii][0]
            assignment_dict['X%s' % i] = x1
            if log == False:
                result = nxor(yi, x1, xii, 'u') * X(x1)
            else:
                result = nxor(yi, x1, xii, 'u') + X(x1)
        else:
            xi = argmax_dict['X%s' % i][yi, xii][0]
            assignment_dict['X%s' % i] = xi
            if log == False:
                result = nxor(yi, xi, xii, 'u') * X(xi) * myi_left(i, xi, argmax_dict)
            else:
                result = nxor(yi, xi, xii, 'u') + X(xi) + myi_left(i, xi, argmax_dict)

        return result

    def my1(x2, y2, argmax_dict):
        z1 = int(z[0])
        y1 = argmax_dict['Y1'][y2, x2][0]
        assignment_dict['Y1'] = y1
        if log == False:
            result = mxi_left(1, y1, x2, argmax_dict) * nxor(z1, y1, y2, 'l')
        else:
            result = mxi_left(1, y1, x2, argmax_dict) + nxor(z1, y1, y2, 'l')
        return result

    def myi_left(i, xii, argmax_dict):
        i -= 1
        zi1 = int(z[i - 2])
        zi2 = int(z[i - 3])
        if i == 2:
            y2 = argmax_dict['Y2'][xii][0]
            assignment_dict['Y%s' % i] = y2
            result = mxi_left(i, y2, xii, argmax_dict)
        else:
            yi = argmax_dict['Y%s' % i][xii][0]
            assignment_dict['Y%s' % i] = yi
            if log == False:
                result = (mxi_left(i, yi, xii, argmax_dict) * nxor(zi1, yi, zi2, 'l'))
            else:
                result = (mxi_left(i, yi, xii, argmax_dict) + nxor(zi1, yi, zi2, 'l'))

        return result

    import numpy as np

    # infer from Zs
    L = len(z)
    i = L + 2

    assignment_dict = {}

    x_L2 = argmax_dict['X%s' % i][0][0]
    assignment_dict['X%s' % i] = x_L2
    if log == False:
        results = X(x_L2) * myi_left(i, x_L2, argmax_dict)
    else:
        results = X(x_L2) + myi_left(i, x_L2, argmax_dict)

    return results, assignment_dict
