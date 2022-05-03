import pandas as pd
import numpy as np

def X(x, q):
    # return input probability
    return np.log(q) if x == 1 else np.log(1 - q)


def nxor(out, in1, in2, p):
    # return noisy XOR probabilities
    if in1 == 1 or in2 == 1:
        return np.log(1 - p) if out == 1 else np.log(p)
    else:
        return np.log(p) if out == 1 else np.log(1 - p)


def generate_clique_dict(z, py_gate, pz_gate, q):
    # create dictionary to store cliques as dictionaries within dictionary
    L = len(z)
    clique_dict = [dict() for i in range(L + 2)]

    # clique zero
    z1 = int(z[0])
    for y1, y2 in [(a, b) for a in [0, 1] for b in [0, 1]]:
        clique_dict[0][f"Y1={y1}, Y2={y2}"] = nxor(z1, y1, y2, pz_gate)

    # clique 1 & 2:
    for xi, yi, xii in [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]:
        prob = X(xi, q) + nxor(yi, xi, xii, py_gate)
        clique_dict[1][f"X1={xi}, Y1={yi}, X2={xii}"] = prob
        clique_dict[2][f"X2={xi}, Y2={yi}, X3={xii}"] = prob

    # clique 3 to L
    for j in range(3, L + 1):
        zi1 = int(z[j - 2])
        zi2 = int(z[j - 3])
        for xi, xii, yi in [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]:
            clique_dict[j][f"X{j}={xi}, Y{j}={yi}, X{j + 1}={xii}"] = X(xi, q) + nxor(yi, xi, xii, py_gate) \
                                                                      + nxor(zi1, zi2, yi, pz_gate)

    # clique L + 1:
    zi1 = int(z[-1])
    zi2 = int(z[-2])
    for xi, xii, yi in [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]:
        clique_dict[L + 1][f"X{L + 1}={xi}, Y{L + 1}={yi}, X{L + 2}={xii}"] = X(xi, q) + X(xii, q) \
                                                                              + nxor(yi, xi, xii, py_gate) \
                                                                              + nxor(zi1, zi2, yi, pz_gate)

    return clique_dict


def generate_messages_LR(clique_dict):
    # from clique i to clique ii (i+1), starting from 0 and ending at L
    L = len(clique_dict) - 2
    messages_LR = []

    for i in range(L + 1):
        p0, p1 = [], []
        if i == 0:
            # message 0-1 (y1, y2)
            m1 = np.array([[clique_dict[0][f"Y1=0, Y2=0"], clique_dict[0][f"Y1=0, Y2=1"]],
                           [clique_dict[0][f"Y1=1, Y2=0"], clique_dict[0][f"Y1=1, Y2=1"]]])
            messages_LR.append(m1)
        elif i == 1:
            # message 1-2 (x2, y2)
            m = [[], [], [], []]
            mii_maxs = []

            m2 = np.zeros((2, 2))
            for x1, y1 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                m[0].append(clique_dict[1][f"X1={x1}, Y1={y1}, X2=0"] + m1[y1, 0])
                m[1].append(clique_dict[1][f"X1={x1}, Y1={y1}, X2=0"] + m1[y1, 1])
                m[2].append(clique_dict[1][f"X1={x1}, Y1={y1}, X2=1"] + m1[y1, 0])
                m[3].append(clique_dict[1][f"X1={x1}, Y1={y1}, X2=1"] + m1[y1, 1])

            m = np.array(m)
            for i, mii in enumerate(m):
                mii_maxs.append(np.max(mii))
                mii -= mii_maxs[i]

            m2[0, 0] = mii_maxs[0] + np.log(np.sum(np.exp(m[0])))
            m2[0, 1] = mii_maxs[1] + np.log(np.sum(np.exp(m[1])))
            m2[1, 0] = mii_maxs[2] + np.log(np.sum(np.exp(m[2])))
            m2[1, 1] = mii_maxs[3] + np.log(np.sum(np.exp(m[3])))
            messages_LR.append(m2)
        elif i == 2:
            # message 2-3 (x3)
            for x2, y2 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                p0.append(clique_dict[2][f"X2={x2}, Y2={y2}, X3=0"] + m2[x2, y2])
                p1.append(clique_dict[2][f"X2={x2}, Y2={y2}, X3=1"] + m2[x2, y2])

            p0, p1 = np.array(p0), np.array(p1)
            p0max, p1max = np.max(p0), np.max(p1)
            p0 -= p0max
            p1 -= p1max
            p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))

            messages_LR.append(np.array([p0, p1]))
        else:
            # message i-ii (xii)
            for xi, yi in [(a, b) for a in [0, 1] for b in [0, 1]]:
                p0.append(clique_dict[i][f"X{i}={xi}, Y{i}={yi}, X{i + 1}=0"] + messages_LR[-1][xi])
                p1.append(clique_dict[i][f"X{i}={xi}, Y{i}={yi}, X{i + 1}=1"] + messages_LR[-1][xi])

            p0, p1 = np.array(p0), np.array(p1)
            p0max, p1max = np.max(p0), np.max(p1)
            p0 -= p0max
            p1 -= p1max
            p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))

            messages_LR.append(np.array([p0, p1]))

    return messages_LR


def generate_messages_RL(clique_dict):
    # from clique i to clique i-1, starting from L+1 ending at 1
    L = len(clique_dict) - 2
    messages_RL = []

    for i in range(L + 1, 0, -1):
        p0, p1 = [], []
        if i == L + 1:
            # message L+1 - L (xL+1)
            for xl2, yl1 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                p0.append(clique_dict[L + 1][f"X{L + 1}=0, Y{L + 1}={yl1}, X{L + 2}={xl2}"])
                p1.append(clique_dict[L + 1][f"X{L + 1}=1, Y{L + 1}={yl1}, X{L + 2}={xl2}"])

            p0, p1 = np.array(p0), np.array(p1)
            p0max, p1max = np.max(p0), np.max(p1)
            p0 -= p0max
            p1 -= p1max
            p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))

            messages_RL.append(np.array([p0, p1]))
        elif i >= 3:
            # message i - i-1 (xi)
            for xii, yi in [(a, b) for a in [0, 1] for b in [0, 1]]:
                p0.append(clique_dict[i][f"X{i}=0, Y{i}={yi}, X{i + 1}={xii}"] + messages_RL[-1][xii])
                p1.append(clique_dict[i][f"X{i}=1, Y{i}={yi}, X{i + 1}={xii}"] + messages_RL[-1][xii])

            p0, p1 = np.array(p0), np.array(p1)
            p0max, p1max = np.max(p0), np.max(p1)
            p0 -= p0max
            p1 -= p1max
            p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))

            messages_RL.append(np.array([p0, p1]))
        elif i == 2:
            # message 2 - 1 (x2, y2)
            m = [[], [], [], []]
            mii_maxs = []

            m2 = np.zeros((2, 2))
            for x3 in [0, 1]:
                m[0].append(clique_dict[2][f"X2=0, Y2=0, X3={x3}"] + messages_RL[-1][x3])
                m[1].append(clique_dict[2][f"X2=0, Y2=1, X3={x3}"] + messages_RL[-1][x3])
                m[2].append(clique_dict[2][f"X2=1, Y2=0, X3={x3}"] + messages_RL[-1][x3])
                m[3].append(clique_dict[2][f"X2=1, Y2=1, X3={x3}"] + messages_RL[-1][x3])

            m = np.array(m)
            for i, mii in enumerate(m):
                mii_maxs.append(np.max(mii))
                mii -= mii_maxs[i]

            m2[0, 0] = mii_maxs[0] + np.log(np.sum(np.exp(m[0])))
            m2[0, 1] = mii_maxs[1] + np.log(np.sum(np.exp(m[1])))
            m2[1, 0] = mii_maxs[2] + np.log(np.sum(np.exp(m[2])))
            m2[1, 1] = mii_maxs[3] + np.log(np.sum(np.exp(m[3])))
            messages_RL.append(m2)
        elif i == 1:
            m = [[], [], [], []]
            mii_maxs = []

            m1 = np.zeros((2, 2))
            # message 1 - 0 (y1, y2)
            for x1, x2 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                m[0].append(clique_dict[1][f"X1={x1}, Y1=0, X2={x2}"] + m2[x2, 0])
                m[1].append(clique_dict[1][f"X1={x1}, Y1=0, X2={x2}"] + m2[x2, 1])
                m[2].append(clique_dict[1][f"X1={x1}, Y1=1, X2={x2}"] + m2[x2, 0])
                m[3].append(clique_dict[1][f"X1={x1}, Y1=1, X2={x2}"] + m2[x2, 1])

            m = np.array(m)
            for i, mii in enumerate(m):
                mii_maxs.append(np.max(mii))
                mii -= mii_maxs[i]

            m1[0, 0] = mii_maxs[0] + np.log(np.sum(np.exp(m[0])))
            m1[0, 1] = mii_maxs[1] + np.log(np.sum(np.exp(m[1])))
            m1[1, 0] = mii_maxs[2] + np.log(np.sum(np.exp(m[2])))
            m1[1, 1] = mii_maxs[3] + np.log(np.sum(np.exp(m[3])))
            messages_RL.append(m1)

    return messages_RL