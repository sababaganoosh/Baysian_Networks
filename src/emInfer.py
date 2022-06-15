# main code q3 c&d

import pandas as pd
import numpy as np

from message_passing import X, nxor, generate_clique_dict, generate_messages_LR, generate_messages_RL


def emInfer(z, py_gate, pz_gate, q, same_p, init=False):
    def likelihood_and_parameters(z, py_gate, pz_gate, q, same_p):

        # identical to hw3 q4 from here

        # def X(x, q):
        #     # return input probability
        #     return np.log(q) if x == 1 else np.log(1 - q)
        #
        # def nxor(out, in1, in2, p):
        #     # return noisy XOR probabilities
        #     if in1 == 1 or in2 == 1:
        #         return np.log(1 - p) if out == 1 else np.log(p)
        #     else:
        #         return np.log(p) if out == 1 else np.log(1 - p)
        #
        # def generate_clique_dict(z, py_gate, pz_gate, q):
        #     # create dictionary to store cliques as dictionaries within dictionary
        #     L = len(z)
        #     clique_dict = [dict() for i in range(L + 2)]
        #
        #     # clique zero
        #     z1 = int(z[0])
        #     for y1, y2 in [(a, b) for a in [0, 1] for b in [0, 1]]:
        #         clique_dict[0][f"Y1={y1}, Y2={y2}"] = nxor(z1, y1, y2, pz_gate)
        #
        #     # clique 1 & 2:
        #     for xi, yi, xii in [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]:
        #         prob = X(xi, q) + nxor(yi, xi, xii, py_gate)
        #         clique_dict[1][f"X1={xi}, Y1={yi}, X2={xii}"] = prob
        #         clique_dict[2][f"X2={xi}, Y2={yi}, X3={xii}"] = prob
        #
        #     # clique 3 to L
        #     for j in range(3, L + 1):
        #         zi1 = int(z[j - 2])
        #         zi2 = int(z[j - 3])
        #         for xi, xii, yi in [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]:
        #             clique_dict[j][f"X{j}={xi}, Y{j}={yi}, X{j + 1}={xii}"] = X(xi, q) + nxor(yi, xi, xii, py_gate) \
        #                                                                       + nxor(zi1, zi2, yi, pz_gate)
        #
        #     # clique L + 1:
        #     zi1 = int(z[-1])
        #     zi2 = int(z[-2])
        #     for xi, xii, yi in [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]:
        #         clique_dict[L + 1][f"X{L + 1}={xi}, Y{L + 1}={yi}, X{L + 2}={xii}"] = X(xi, q) + X(xii, q) \
        #                                                                               + nxor(yi, xi, xii, py_gate) \
        #                                                                               + nxor(zi1, zi2, yi, pz_gate)
        #
        #     return clique_dict
        #
        # def generate_messages_LR(clique_dict):
        #     # from clique i to clique ii (i+1), starting from 0 and ending at L
        #     L = len(clique_dict) - 2
        #     messages_LR = []
        #
        #     for i in range(L + 1):
        #         p0, p1 = [], []
        #         if i == 0:
        #             # message 0-1 (y1, y2)
        #             m1 = np.array([[clique_dict[0][f"Y1=0, Y2=0"], clique_dict[0][f"Y1=0, Y2=1"]],
        #                            [clique_dict[0][f"Y1=1, Y2=0"], clique_dict[0][f"Y1=1, Y2=1"]]])
        #             messages_LR.append(m1)
        #         elif i == 1:
        #             # message 1-2 (x2, y2)
        #             m = [[], [], [], []]
        #             mii_maxs = []
        #
        #             m2 = np.zeros((2, 2))
        #             for x1, y1 in [(a, b) for a in [0, 1] for b in [0, 1]]:
        #                 m[0].append(clique_dict[1][f"X1={x1}, Y1={y1}, X2=0"] + m1[y1, 0])
        #                 m[1].append(clique_dict[1][f"X1={x1}, Y1={y1}, X2=0"] + m1[y1, 1])
        #                 m[2].append(clique_dict[1][f"X1={x1}, Y1={y1}, X2=1"] + m1[y1, 0])
        #                 m[3].append(clique_dict[1][f"X1={x1}, Y1={y1}, X2=1"] + m1[y1, 1])
        #
        #             m = np.array(m)
        #             for i, mii in enumerate(m):
        #                 mii_maxs.append(np.max(mii))
        #                 mii -= mii_maxs[i]
        #
        #             m2[0, 0] = mii_maxs[0] + np.log(np.sum(np.exp(m[0])))
        #             m2[0, 1] = mii_maxs[1] + np.log(np.sum(np.exp(m[1])))
        #             m2[1, 0] = mii_maxs[2] + np.log(np.sum(np.exp(m[2])))
        #             m2[1, 1] = mii_maxs[3] + np.log(np.sum(np.exp(m[3])))
        #             messages_LR.append(m2)
        #         elif i == 2:
        #             # message 2-3 (x3)
        #             for x2, y2 in [(a, b) for a in [0, 1] for b in [0, 1]]:
        #                 p0.append(clique_dict[2][f"X2={x2}, Y2={y2}, X3=0"] + m2[x2, y2])
        #                 p1.append(clique_dict[2][f"X2={x2}, Y2={y2}, X3=1"] + m2[x2, y2])
        #
        #             p0, p1 = np.array(p0), np.array(p1)
        #             p0max, p1max = np.max(p0), np.max(p1)
        #             p0 -= p0max
        #             p1 -= p1max
        #             p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
        #
        #             messages_LR.append(np.array([p0, p1]))
        #         else:
        #             # message i-ii (xii)
        #             for xi, yi in [(a, b) for a in [0, 1] for b in [0, 1]]:
        #                 p0.append(clique_dict[i][f"X{i}={xi}, Y{i}={yi}, X{i + 1}=0"] + messages_LR[-1][xi])
        #                 p1.append(clique_dict[i][f"X{i}={xi}, Y{i}={yi}, X{i + 1}=1"] + messages_LR[-1][xi])
        #
        #             p0, p1 = np.array(p0), np.array(p1)
        #             p0max, p1max = np.max(p0), np.max(p1)
        #             p0 -= p0max
        #             p1 -= p1max
        #             p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
        #
        #             messages_LR.append(np.array([p0, p1]))
        #
        #     return messages_LR
        #
        # def generate_messages_RL(clique_dict):
        #     # from clique i to clique i-1, starting from L+1 ending at 1
        #     L = len(clique_dict) - 2
        #     messages_RL = []
        #
        #     for i in range(L + 1, 0, -1):
        #         p0, p1 = [], []
        #         if i == L + 1:
        #             # message L+1 - L (xL+1)
        #             for xl2, yl1 in [(a, b) for a in [0, 1] for b in [0, 1]]:
        #                 p0.append(clique_dict[L + 1][f"X{L + 1}=0, Y{L + 1}={yl1}, X{L + 2}={xl2}"])
        #                 p1.append(clique_dict[L + 1][f"X{L + 1}=1, Y{L + 1}={yl1}, X{L + 2}={xl2}"])
        #
        #             p0, p1 = np.array(p0), np.array(p1)
        #             p0max, p1max = np.max(p0), np.max(p1)
        #             p0 -= p0max
        #             p1 -= p1max
        #             p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
        #
        #             messages_RL.append(np.array([p0, p1]))
        #         elif i >= 3:
        #             # message i - i-1 (xi)
        #             for xii, yi in [(a, b) for a in [0, 1] for b in [0, 1]]:
        #                 p0.append(clique_dict[i][f"X{i}=0, Y{i}={yi}, X{i + 1}={xii}"] + messages_RL[-1][xii])
        #                 p1.append(clique_dict[i][f"X{i}=1, Y{i}={yi}, X{i + 1}={xii}"] + messages_RL[-1][xii])
        #
        #             p0, p1 = np.array(p0), np.array(p1)
        #             p0max, p1max = np.max(p0), np.max(p1)
        #             p0 -= p0max
        #             p1 -= p1max
        #             p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
        #
        #             messages_RL.append(np.array([p0, p1]))
        #         elif i == 2:
        #             # message 2 - 1 (x2, y2)
        #             m = [[], [], [], []]
        #             mii_maxs = []
        #
        #             m2 = np.zeros((2, 2))
        #             for x3 in [0, 1]:
        #                 m[0].append(clique_dict[2][f"X2=0, Y2=0, X3={x3}"] + messages_RL[-1][x3])
        #                 m[1].append(clique_dict[2][f"X2=0, Y2=1, X3={x3}"] + messages_RL[-1][x3])
        #                 m[2].append(clique_dict[2][f"X2=1, Y2=0, X3={x3}"] + messages_RL[-1][x3])
        #                 m[3].append(clique_dict[2][f"X2=1, Y2=1, X3={x3}"] + messages_RL[-1][x3])
        #
        #             m = np.array(m)
        #             for i, mii in enumerate(m):
        #                 mii_maxs.append(np.max(mii))
        #                 mii -= mii_maxs[i]
        #
        #             m2[0, 0] = mii_maxs[0] + np.log(np.sum(np.exp(m[0])))
        #             m2[0, 1] = mii_maxs[1] + np.log(np.sum(np.exp(m[1])))
        #             m2[1, 0] = mii_maxs[2] + np.log(np.sum(np.exp(m[2])))
        #             m2[1, 1] = mii_maxs[3] + np.log(np.sum(np.exp(m[3])))
        #             messages_RL.append(m2)
        #         elif i == 1:
        #             m = [[], [], [], []]
        #             mii_maxs = []
        #
        #             m1 = np.zeros((2, 2))
        #             # message 1 - 0 (y1, y2)
        #             for x1, x2 in [(a, b) for a in [0, 1] for b in [0, 1]]:
        #                 m[0].append(clique_dict[1][f"X1={x1}, Y1=0, X2={x2}"] + m2[x2, 0])
        #                 m[1].append(clique_dict[1][f"X1={x1}, Y1=0, X2={x2}"] + m2[x2, 1])
        #                 m[2].append(clique_dict[1][f"X1={x1}, Y1=1, X2={x2}"] + m2[x2, 0])
        #                 m[3].append(clique_dict[1][f"X1={x1}, Y1=1, X2={x2}"] + m2[x2, 1])
        #
        #             m = np.array(m)
        #             for i, mii in enumerate(m):
        #                 mii_maxs.append(np.max(mii))
        #                 mii -= mii_maxs[i]
        #
        #             m1[0, 0] = mii_maxs[0] + np.log(np.sum(np.exp(m[0])))
        #             m1[0, 1] = mii_maxs[1] + np.log(np.sum(np.exp(m[1])))
        #             m1[1, 0] = mii_maxs[2] + np.log(np.sum(np.exp(m[2])))
        #             m1[1, 1] = mii_maxs[3] + np.log(np.sum(np.exp(m[3])))
        #             messages_RL.append(m1)
        #
        #     return messages_RL

        # new hw4 q3 code from here
        # get variable posterior probabilities:
        def get_var_likelihood(var, messages_LR, messages_RL):
            i = int(var[1:])
            if var[0] == "Z":
                if i == 1:
                    ml = None
                    mr = messages_RL[-1]
                else:
                    ml = messages_LR[i]
                    mr = messages_RL[-i - 2] if i <= L - 1 else None
            else:
                ml = messages_LR[i - 1] if i < L + 1 else messages_LR[-1]
                mr = messages_RL[L - i] if i <= L else None

            if var[0] == "X":
                p0, p1 = [], []
                if i == 1:
                    for x2, y1, y2 in [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]:
                        p0.append(clique_dict[i][f"X1=0, Y1={y1}, X2={x2}"] + ml[y1, y2] + mr[x2, y2])
                        p1.append(clique_dict[i][f"X1=1, Y1={y1}, X2={x2}"] + ml[y1, y2] + mr[x2, y2])

                    p0, p1 = np.array(p0), np.array(p1)
                    p0max, p1max = np.max(p0), np.max(p1)
                    p0 -= p0max
                    p1 -= p1max
                    p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
                elif i == 2:
                    for x3, y2 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                        p0.append(clique_dict[i][f"X2=0, Y2={y2}, X3={x3}"] + ml[0, y2] + mr[x3])
                        p1.append(clique_dict[i][f"X2=1, Y2={y2}, X3={x3}"] + ml[1, y2] + mr[x3])

                    p0, p1 = np.array(p0), np.array(p1)
                    p0max, p1max = np.max(p0), np.max(p1)
                    p0 -= p0max
                    p1 -= p1max
                    p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
                elif i < L + 1:
                    for xii, yi in [(a, b) for a in [0, 1] for b in [0, 1]]:
                        p0.append(clique_dict[i][f"X{i}=0, Y{i}={yi}, X{i + 1}={xii}"] + ml[0] + mr[xii])
                        p1.append(clique_dict[i][f"X{i}=1, Y{i}={yi}, X{i + 1}={xii}"] + ml[1] + mr[xii])

                    p0, p1 = np.array(p0), np.array(p1)
                    p0max, p1max = np.max(p0), np.max(p1)
                    p0 -= p0max
                    p1 -= p1max
                    p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
                elif i == L + 1:
                    for xl2, yl1 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                        p0.append(clique_dict[i][f"X{L + 1}=0, Y{L + 1}={yl1}, X{L + 2}={xl2}"] + ml[0])
                        p1.append(clique_dict[i][f"X{L + 1}=1, Y{L + 1}={yl1}, X{L + 2}={xl2}"] + ml[1])

                    p0, p1 = np.array(p0), np.array(p1)
                    p0max, p1max = np.max(p0), np.max(p1)
                    p0 -= p0max
                    p1 -= p1max
                    p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
                elif i == L + 2:
                    for xl1, yl1 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                        p0.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}={yl1}, X{L + 2}=0"] + ml[xl1])
                        p1.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}={yl1}, X{L + 2}=1"] + ml[xl1])

                    p0, p1 = np.array(p0), np.array(p1)
                    p0max, p1max = np.max(p0), np.max(p1)
                    p0 -= p0max
                    p1 -= p1max
                    p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))

                marg_prob = np.array([p0, p1])

            elif var[0] == "Y":
                py, pyc = [], []
                if i == 1:
                    for x1, x2, y2 in [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]:
                        if x1 == 1 or x2 == 1:
                            py.append(clique_dict[i][f"X1={x1}, Y1=0, X2={x2}"] + ml[0, y2] + mr[x2, y2])
                            pyc.append(clique_dict[i][f"X1={x1}, Y1=1, X2={x2}"] + ml[1, y2] + mr[x2, y2])
                        else:
                            pyc.append(clique_dict[i][f"X1={x1}, Y1=0, X2={x2}"] + ml[0, y2] + mr[x2, y2])
                            py.append(clique_dict[i][f"X1={x1}, Y1=1, X2={x2}"] + ml[1, y2] + mr[x2, y2])

                    py, pyc = np.array(py), np.array(pyc)
                    pymax, pycmax = np.max(py), np.max(pyc)
                    py -= pymax
                    pyc -= pycmax
                    py, pyc = pymax + np.log(np.sum(np.exp(py))), pycmax + np.log(np.sum(np.exp(pyc)))
                elif i == 2:
                    for x2, x3 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                        if x2 == 1 or x3 == 1:
                            py.append(clique_dict[i][f"X2={x2}, Y2=0, X3={x3}"] + ml[x2, 0] + mr[x3])
                            pyc.append(clique_dict[i][f"X2={x2}, Y2=1, X3={x3}"] + ml[x2, 1] + mr[x3])
                        else:
                            pyc.append(clique_dict[i][f"X2={x2}, Y2=0, X3={x3}"] + ml[x2, 0] + mr[x3])
                            py.append(clique_dict[i][f"X2={x2}, Y2=1, X3={x3}"] + ml[x2, 1] + mr[x3])

                    py, pyc = np.array(py), np.array(pyc)
                    pymax, pycmax = np.max(py), np.max(pyc)
                    py -= pymax
                    pyc -= pycmax
                    py, pyc = pymax + np.log(np.sum(np.exp(py))), pycmax + np.log(np.sum(np.exp(pyc)))
                elif i < L + 1:
                    for xi, xii in [(a, b) for a in [0, 1] for b in [0, 1]]:
                        if xi == 1 or xii == 1:
                            py.append(clique_dict[i][f"X{i}={xi}, Y{i}=0, X{i + 1}={xii}"] + ml[xi] + mr[xii])
                            pyc.append(clique_dict[i][f"X{i}={xi}, Y{i}=1, X{i + 1}={xii}"] + ml[xi] + mr[xii])
                        else:
                            pyc.append(clique_dict[i][f"X{i}={xi}, Y{i}=0, X{i + 1}={xii}"] + ml[xi] + mr[xii])
                            py.append(clique_dict[i][f"X{i}={xi}, Y{i}=1, X{i + 1}={xii}"] + ml[xi] + mr[xii])

                    py, pyc = np.array(py), np.array(pyc)
                    pymax, pycmax = np.max(py), np.max(pyc)
                    py -= pymax
                    pyc -= pycmax
                    py, pyc = pymax + np.log(np.sum(np.exp(py))), pycmax + np.log(np.sum(np.exp(pyc)))
                elif i == L + 1:
                    for xl1, xl2 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                        if xl1 == 1 or xl2 == 1:
                            py.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=0, X{L + 2}={xl2}"] + ml[xl1])
                            pyc.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=1, X{L + 2}={xl2}"] + ml[xl1])
                        else:
                            pyc.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=0, X{L + 2}={xl2}"] + ml[xl1])
                            py.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=1, X{L + 2}={xl2}"] + ml[xl1])

                    py, pyc = np.array(py), np.array(pyc)
                    pymax, pycmax = np.max(py), np.max(pyc)
                    py -= pymax
                    pyc -= pycmax
                    py, pyc = pymax + np.log(np.sum(np.exp(py))), pycmax + np.log(np.sum(np.exp(pyc)))

                marg_prob = np.array([py, pyc])

            elif var[0] == "Z":
                pz, pzc = [], []
                zi = int(z[i - 1])
                small_p, big_p = np.log(.00001), np.log(.99999)
                #                 small_p, big_p = 0, 1

                if i > 1:
                    zi_minus1 = int(z[i - 2])

                if i == 1:
                    if zi == 1:
                        pz.append(clique_dict[0][f"Y1=0, Y2=0"] + mr[0, 0])
                        pzc.append(clique_dict[0][f"Y1=1, Y2=0"] + mr[1, 0])
                        pzc.append(clique_dict[0][f"Y1=0, Y2=1"] + mr[0, 1])
                        pzc.append(clique_dict[0][f"Y1=1, Y2={y2}"] + mr[1, 1])
                    else:
                        pzc.append(clique_dict[0][f"Y1=0, Y2=0"] + mr[0, 0])
                        pz.append(clique_dict[0][f"Y1=1, Y2=0"] + mr[1, 0])
                        pz.append(clique_dict[0][f"Y1=0, Y2=1"] + mr[0, 1])
                        pz.append(clique_dict[0][f"Y1=1, Y2=1"] + mr[1, 1])

                    if len(pz) == 0:
                        pz, pzc = small_p, big_p
                    elif len(pzc) == 0:
                        pz, pzc = big_p, small_p
                    else:
                        pz, pzc = np.array(pz), np.array(pzc)
                        pzmax, pzcmax = np.max(pz), np.max(pzc)
                        pz -= pzmax
                        pzc -= pzcmax
                        pz, pzc = pzmax + np.log(np.sum(np.exp(pz))), pzcmax + np.log(np.sum(np.exp(pzc)))
                elif i == 2:
                    for x3, x4 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                        if zi == 1:
                            if zi_minus1 == 1:
                                pzc.append(clique_dict[i + 1][f"X3={x3}, Y3=0, X4={x4}"] + ml[x3] + mr[x4])
                                pzc.append(clique_dict[i + 1][f"X3={x3}, Y3=1, X4={x4}"] + ml[x3] + mr[x4])
                            else:
                                pz.append(clique_dict[i + 1][f"X3={x3}, Y3=0, X4={x4}"] + ml[x3] + mr[x4])
                                pzc.append(clique_dict[i + 1][f"X3={x3}, Y3=1, X4={x4}"] + ml[x3] + mr[x4])

                        else:
                            if zi_minus1 == 1:
                                pz.append(clique_dict[i + 1][f"X3={x3}, Y3=0, X4={x4}"] + ml[x3] + mr[x4])
                                pz.append(clique_dict[i + 1][f"X3={x3}, Y3=1, X4={x4}"] + ml[x3] + mr[x4])
                            else:
                                pzc.append(clique_dict[i + 1][f"X3={x3}, Y3=0, X4={x4}"] + ml[x3] + mr[x4])
                                pz.append(clique_dict[i + 1][f"X3={x3}, Y3=1, X4={x4}"] + ml[x3] + mr[x4])

                    if len(pz) == 0:
                        pz, pzc = small_p, big_p
                    elif len(pzc) == 0:
                        pz, pzc = big_p, small_p
                    else:
                        pz, pzc = np.array(pz), np.array(pzc)
                        pzmax, pzcmax = np.max(pz), np.max(pzc)
                        pz -= pzmax
                        pzc -= pzcmax
                        pz, pzc = pzmax + np.log(np.sum(np.exp(pz))), pzcmax + np.log(np.sum(np.exp(pzc)))
                elif i < L:
                    for xi, xii in [(a, b) for a in [0, 1] for b in [0, 1]]:
                        if zi == 1:
                            if zi_minus1 == 1:
                                pzc.append(
                                    clique_dict[i + 1][f"X{i + 1}={xi}, Y{i + 1}=0, X{i + 2}={xii}"] + ml[xi] + mr[xii])
                                pzc.append(
                                    clique_dict[i + 1][f"X{i + 1}={xi}, Y{i + 1}=1, X{i + 2}={xii}"] + ml[xi] + mr[xii])
                            else:
                                pz.append(
                                    clique_dict[i + 1][f"X{i + 1}={xi}, Y{i + 1}=0, X{i + 2}={xii}"] + ml[xi] + mr[xii])
                                pzc.append(
                                    clique_dict[i + 1][f"X{i + 1}={xi}, Y{i + 1}=1, X{i + 2}={xii}"] + ml[xi] + mr[xii])
                        else:
                            if zi_minus1 == 1:
                                pz.append(
                                    clique_dict[i + 1][f"X{i + 1}={xi}, Y{i + 1}=0, X{i + 2}={xii}"] + ml[xi] + mr[xii])
                                pz.append(
                                    clique_dict[i + 1][f"X{i + 1}={xi}, Y{i + 1}=1, X{i + 2}={xii}"] + ml[xi] + mr[xii])
                            else:
                                pzc.append(
                                    clique_dict[i + 1][f"X{i + 1}={xi}, Y{i + 1}=0, X{i + 2}={xii}"] + ml[xi] + mr[xii])
                                pz.append(
                                    clique_dict[i + 1][f"X{i + 1}={xi}, Y{i + 1}=1, X{i + 2}={xii}"] + ml[xi] + mr[xii])

                    if len(pz) == 0:
                        pz, pzc = small_p, big_p
                    elif len(pzc) == 0:
                        pz, pzc = big_p, small_p
                    else:
                        pz, pzc = np.array(pz), np.array(pzc)
                        pzmax, pzcmax = np.max(pz), np.max(pzc)
                        pz -= pzmax
                        pzc -= pzcmax
                        pz, pzc = pzmax + np.log(np.sum(np.exp(pz))), pzcmax + np.log(np.sum(np.exp(pzc)))

                elif i == L:
                    for xl1, xl2 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                        if zi == 1:
                            if zi_minus1 == 1:
                                pzc.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=0, X{L + 2}={xl2}"] + ml[xl1])
                                pzc.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=1, X{L + 2}={xl2}"] + ml[xl1])
                            else:
                                pz.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=0, X{L + 2}={xl2}"] + ml[xl1])
                                pzc.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=1, X{L + 2}={xl2}"] + ml[xl1])

                        else:
                            if zi_minus1 == 1:
                                pz.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=0, X{L + 2}={xl2}"] + ml[xl1])
                                pz.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=1, X{L + 2}={xl2}"] + ml[xl1])
                            else:
                                pzc.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=0, X{L + 2}={xl2}"] + ml[xl1])
                                pz.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=1, X{L + 2}={xl2}"] + ml[xl1])

                    if len(pz) == 0:
                        pz, pzc = small_p, big_p
                    elif len(pzc) == 0:
                        pz, pzc = big_p, small_p
                    else:
                        pz, pzc = np.array(pz), np.array(pzc)
                        pzmax, pzcmax = np.max(pz), np.max(pzc)
                        pz -= pzmax
                        pzc -= pzcmax
                        pz, pzc = pzmax + np.log(np.sum(np.exp(pz))), pzcmax + np.log(np.sum(np.exp(pzc)))

                marg_prob = np.array([pz, pzc])

            marg_prob_max = np.max(marg_prob)
            marg_prob_sum = marg_prob_max + np.log(np.sum(np.exp(marg_prob - marg_prob_max)))
            return marg_prob - marg_prob_sum

        # main

        L = len(z)
        clique_dict = generate_clique_dict(z, py_gate, pz_gate, q)
        messages_LR = generate_messages_LR(clique_dict)
        messages_RL = generate_messages_RL(clique_dict)

        # calculate likelihood
        likelihood = []
        for y1, y2 in [(a, b) for a in [0, 1] for b in [0, 1]]:
            likelihood.append(clique_dict[0][f"Y1={y1}, Y2={y2}"] + messages_RL[-1][y1, y2])

        likelihood_max = np.max(likelihood)
        likelihood -= likelihood_max
        likelihood = likelihood_max + np.log(np.sum(np.exp(likelihood)))
        # calculate likelihood in reverse and sanity check:
        likelihood2 = []
        for xl1, yl1, xl2 in [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]:
            likelihood2.append(
                clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}={yl1}, X{L + 2}={xl2}"] + messages_LR[-1][xl1])

        likelihood2_max = np.max(likelihood2)
        likelihood2 -= likelihood2_max
        likelihood2 = likelihood2_max + np.log(np.sum(np.exp(likelihood2)))
        if round(likelihood, 5) != round(likelihood2, 5):
            raise ValueError(f"Two likelihoods {likelihood}, {likelihood2} are different!")

        # get posterior probability
        post_X = []
        post_Y = []
        post_Z = []

        for i in range(1, L + 3):
            pxi = get_var_likelihood(f"X{i}", messages_LR, messages_RL)
            post_X.append(np.exp(pxi[1]))
            if i != L + 2:
                pyi = get_var_likelihood(f"Y{i}", messages_LR, messages_RL)
                post_Y.append(np.exp(pyi[0]))
            if i != L + 2 and i != L + 1:
                pzi = get_var_likelihood(f"Z{i}", messages_LR, messages_RL)
                post_Z.append(np.exp(pzi[0]))

        q = sum(post_X) / (L + 2)
        if same_p:
            py_gate = sum(post_Y + post_Z) / (L + L + 1)
            pz_gate = py_gate
        else:
            py_gate, pz_gate = sum(post_Y) / (L + 1), sum(post_Z) / L

        likelihood, py_gate, pz_gate, q = round(likelihood, 5), round(py_gate, 5), round(pz_gate, 5), round(q, 5)

        return clique_dict, messages_LR, messages_RL, likelihood, py_gate, pz_gate, q

    iteration_vals = [[py_gate, pz_gate, q, 0]]
    change = True

    if init:
        clique_dict, messages_LR, messages_RL, likelihood, py_gate, pz_gate, q \
            = likelihood_and_parameters(z, py_gate, pz_gate, q, same_p)

        iteration_vals[-1][-1] = likelihood


    else:
        while change:
            clique_dict, messages_LR, messages_RL, likelihood, py_gate, pz_gate, q \
                = likelihood_and_parameters(z, py_gate, pz_gate, q, same_p)

            if iteration_vals[-1][-1] == likelihood:
                change = False

            iteration_vals[-1][-1] = likelihood
            if change:
                iteration_vals.append([py_gate, pz_gate, q, likelihood])

    df = pd.DataFrame(data=iteration_vals, columns=['Py', 'Pz', 'q', 'log likelihood'])

    return clique_dict, messages_LR, messages_RL, df

# def visualize(iteration_vals):
#     df = pd.DataFrame(data = iteration_vals[1:], columns = ['Py', 'Pz', 'q', 'log likelihood'])
#     df.iloc[np.r_[0:7, -3:0]]
