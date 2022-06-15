import numpy as np

from message_passing import X, nxor, generate_clique_dict, generate_messages_LR, generate_messages_RL

def likelihood(z, p, q):

    # get variable posterior probabilities:
    def get_var_likelihood(var, messages_LR, messages_RL):
        i = int(var[1:])
        ml = messages_LR[i - 1] if i < L + 1 else messages_LR[-1]
        mr = messages_RL[L - i] if i <= L else None
        p0, p1 = [], []
        if var[0] == "X":
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

        elif var[0] == "Y":
            if i == 1:
                for x1, x2, y2 in [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]:
                    p0.append(clique_dict[i][f"X1={x1}, Y1=0, X2={x2}"] + ml[0, y2] + mr[x2, y2])
                    p1.append(clique_dict[i][f"X1={x1}, Y1=1, X2={x2}"] + ml[1, y2] + mr[x2, y2])

                p0, p1 = np.array(p0), np.array(p1)
                p0max, p1max = np.max(p0), np.max(p1)
                p0 -= p0max
                p1 -= p1max
                p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
            elif i == 2:
                for x2, x3 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                    p0.append(clique_dict[i][f"X2={x2}, Y2=0, X3={x3}"] + ml[x2, 0] + mr[x3])
                    p1.append(clique_dict[i][f"X2={x2}, Y2=1, X3={x3}"] + ml[x2, 1] + mr[x3])

                p0, p1 = np.array(p0), np.array(p1)
                p0max, p1max = np.max(p0), np.max(p1)
                p0 -= p0max
                p1 -= p1max
                p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
            elif i < L + 1:
                for xi, xii in [(a, b) for a in [0, 1] for b in [0, 1]]:
                    p0.append(clique_dict[i][f"X{i}={xi}, Y{i}=0, X{i + 1}={xii}"] + ml[xi] + mr[xii])
                    p1.append(clique_dict[i][f"X{i}={xi}, Y{i}=1, X{i + 1}={xii}"] + ml[xi] + mr[xii])

                p0, p1 = np.array(p0), np.array(p1)
                p0max, p1max = np.max(p0), np.max(p1)
                p0 -= p0max
                p1 -= p1max
                p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
            elif i == L + 1:
                for xl1, xl2 in [(a, b) for a in [0, 1] for b in [0, 1]]:
                    p0.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=0, X{L + 2}={xl2}"] + ml[xl1])
                    p1.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}=1, X{L + 2}={xl2}"] + ml[xl1])

                p0, p1 = np.array(p0), np.array(p1)
                p0max, p1max = np.max(p0), np.max(p1)
                p0 -= p0max
                p1 -= p1max
                p0, p1 = p0max + np.log(np.sum(np.exp(p0))), p1max + np.log(np.sum(np.exp(p1)))
        marg_prob = np.array([p0, p1])

        marg_prob_max = np.max(marg_prob)
        marg_prob_sum = marg_prob_max + np.log(np.sum(np.exp(marg_prob - marg_prob_max)))
        return marg_prob - marg_prob_sum

    # main

    L = len(z)
    clique_dict = generate_clique_dict(z, p, p, q)
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
        likelihood2.append(clique_dict[-1][f"X{L + 1}={xl1}, Y{L + 1}={yl1}, X{L + 2}={xl2}"] + messages_LR[-1][xl1])

    likelihood2_max = np.max(likelihood2)
    likelihood2 -= likelihood2_max
    likelihood2 = likelihood2_max + np.log(np.sum(np.exp(likelihood2)))
    if round(likelihood, 5) != round(likelihood2, 5):
        raise ValueError(f"Two likelihoods {likelihood}, {likelihood2} are different!")

    # get posterior probability
    post_X = []
    post_Y = []
    for i in range(1, L + 3):
        pxi = get_var_likelihood(f"X{i}", messages_LR, messages_RL)
        post_X.append(round(np.exp(pxi[1]) * 100))
        if i != L + 2:
            pyi = get_var_likelihood(f"Y{i}", messages_LR, messages_RL)
            post_Y.append(round(np.exp(pyi[1]) * 100))

    def visualize(likelihood, Xs, Ys):
        L = len(z)
        s1 = 'Xi:   '
        for i in range(L + 1):
            s1 += str(Xs[i])
            if len(str(Xs[i + 1])) == 2:
                s1 += " "
            else:
                s1 += "  "
        s1 += str(Xs[-1])
        s2 = "    "
        for j in range(len(str(Xs[0]))):
            s2 += " "
        for i in range(L + 1):
            s2 += "  "
            s2 += "V"
        s3 = 'Yi:    '
        for i in range(L):
            s3 += str(Ys[i])
            if len(str(Ys[i + 1])) == 2:
                s3 += " "
            else:
                s3 += "  "
        s3 += str(Ys[-1])

        s4 = "       "
        for j in range(len(str(Ys[0]))):
            s4 += " "
        s4 += 'V'
        for i in range(1, L):
            s4 += "  /"
        s5 = "Zi:    "
        for j in range(len(str(Ys[0]))):
            s5 += " "
        for i in range(L - 1):
            s5 += z[i]
            s5 += "--"
        s5 += z[-1]

        s = f'Optimal assignment to hidden RVs:\n{s1}\n{s2}\n{s3}\n{s4}\n{s5}'
        print(f'Observed Sequence Z = {z}')
        print(f'Parameters: p = {p}, q = {q}')
        print(s)
        print(f'Data Log Likelihood ln(P(Z)) = {likelihood}')

    visualize(likelihood, post_X, post_Y)
    return clique_dict, messages_LR, messages_RL, likelihood, post_X, post_Y
