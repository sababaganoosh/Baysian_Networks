# main code q2 a&b
import pandas as pd
from inference import inference
from reconstruct import reconstruct
from visualize import visualize


def maxProbInfer(z, py0, pz0, q0, same_p=False):
    def get_assignment(z, py, pz, q, log=True):
        # get maximum probability assignment using hw3 code
        # print(z, py, pz, q, log)
        inf_dict = inference(z, py, pz, q, log)
        logprob, assignment = reconstruct(z, py, pz, q, inf_dict, log)
        # visualize(logprob, assignment)
        return logprob, assignment

    def get_params(z, assignment, same_p):
        # sufficient statistics
        s1, s2, s3 = 0, 0, 0
        L = len(z)
        for i in range(1, L + 3):
            if assignment[f'X{i}'] == 1:
                s1 += 1
            if i <= L + 1:
                if (assignment[f'X{i}'] == 1 or assignment[f'X{i + 1}'] == 1):
                    s2 += 1 if assignment[f'Y{i}'] == 0 else 0
                else:
                    s2 += 1 if assignment[f'Y{i}'] == 1 else 0
            if i == 1:
                if (assignment[f'Y1'] == 1 or assignment[f'Y2'] == 1):
                    s3 += 1 if int(z[0]) == 0 else 0
                else:
                    s3 += 1 if int(z[0]) == 1 else 0
            if i > 1 and i <= L:
                if (assignment[f'Y{i + 1}'] == 1 or int(z[i - 2]) == 1):
                    s3 += 1 if int(z[i - 1]) == 0 else 0
                else:
                    s3 += 1 if int(z[i - 1]) == 1 else 0
        # print('Sufficient stats:', s1, s2, s3)

        # ensure numerical stability
        if s1 == 0:
            s1 = 0.00001
        elif s1 == L + 2:
            s1 = L + 2 - 0.00001
        if s2 == 0:
            s2 = 0.00001
        elif s2 == L + 1:
            s2 = L + 1 - 0.00001
        if s3 == 0:
            s3 = 0.00001
        elif s3 == L:
            s1 = L - 0.00001

        # calculate parameter based on MLEs:
        q = s1 / (L + 2)
        if same_p:
            # for question 2b
            py = (s2 + s3) / (2 * L + 1)
            pz = py
        else:
            py = s2 / (L + 1)
            pz = s3 / L
        # print(py, pz, q)
        # log probabilities
        return py, pz, q

        # main function

    # store all params and probs in lists
    Q = [q0]
    Py = [py0]
    Pz = [pz0]
    LogProb = []

    # main loop, stop when the parameters stop changing
    previous_params = (0, 0, 0)
    maxIter = 20
    Iter = 0
    while (Py[-1], Pz[-1], Q[-1]) != previous_params:

        # for stopping condition
        previous_params = (Py[-1], Pz[-1], Q[-1])
        py, pz, q = previous_params

        # get assignment, then get params
        log_prob, assignment = get_assignment(z, py, pz, q, log=True)
        py, pz, q = get_params(z, assignment, same_p)

        # append to list
        Q.append(q)
        Py.append(py)
        Pz.append(pz)
        LogProb.append(log_prob)

        # just so it doesn't run too long
        Iter += 1
        if Iter > maxIter:
            break

    # print result
    return pd.DataFrame({'Py': Py[:-1], 'Pz': Pz[:-1], 'q': Q[-1], 'log max prob': LogProb})
#     print("Py        |Pz         |q          |log max prob")
#     for q, py, pz, pr in zip(Q[:-1], Py[:-1], Pz[:-1], LogProb):
#         print(f"{py:.3f}     |{pz:.3f}      |{q:.3f}      |{pr:.3f}")
