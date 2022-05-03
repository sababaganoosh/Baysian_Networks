def visualize(z, py, pz, q, log, prob, assignment):
    L = len(z)
    s1= 'Xi:   '
    for i in range(1, L+3):
        s1 += str(assignment['X%s' % i])
        s1 += " "
    s2 = "      "
    for i in range(1, L+2):
        s2 += " V"
    s3 = 'Yi:   '
    for i in range(1, L+2):
        s3 += " "
        s3 += str(assignment['Y%s' % i])
    s4 = "        V"
    for i in range(1, L):
        s4 += " /"
    s5 = "Zi:     "
    for i in range(0, L-1):
        s5 += z[i]
        s5 += "-"
    s5 += z[-1]

    s = f'Optimal assignment to hidden RVs:\n{s1}\n{s2}\n{s3}\n{s4}\n{s5}'
    print(f'Observed Sequence Z = {z}')
    print(f'Parameters: py = {py}, pz = {pz}, q = {q}')
    print(s)
    if log == False:
        print(f'Joint Log Probability ln(P(X,Y,Z)) = {prob}')
    else:
        print(f'Joint Probability P(X,Y,Z) = {prob}')