
import numpy as np


def needleman_wunsch(x, y, match=1, mismatch=1, gap=1, fillchar='-'):
    '''
    Needleman-Wunsch sequence alignement algorithm, used to align differing protocols
    
    :param x: first sequence (string or list)
    :param y: second sequence (string or list)
    :param match: value added to local score in case of matched sequences
    :param mismatch: value subtracted from local score in case of mismatched sequences
    :param gap: value substracted from local score in case of shift in either sequence
    :param fillchar: character used to fill gaps
    :return: tuple of aligned sequences (same type as inputs)
    '''
    # Get input size
    nx, ny = len(x), len(y)
    # Define matrix of optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:, 0] = np.linspace(0, -nx, nx + 1)
    F[0, :] = np.linspace(0, -ny, ny + 1)
    # Define pointers matrix to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4
    # Define temporary scores vector
    t = np.zeros(3)
    
    # Loop through both sequences
    for i in range(nx):
        for j in range(ny):
            # Compute local score based solely on match/mismatch
            if x[i] == y[j]:
                t[0] = F[i, j] + match
            else:
                t[0] = F[i, j] - mismatch
            # Compute "shifted" scores based on neighboring score + gap penalty for shifting
            # either first or second sequence
            t[1] = F[i, j + 1] - gap
            t[2] = F[i + 1, j] - gap
            # Get max local score and assign it in score matrix
            tmax = np.max(t)
            F[i + 1, j + 1] = tmax
            # Add specific value to tracer matrix depending on where max was found
            # These values are chosen such that their sum can be used to trace back 
            # what was added
            # If no shift yields local max, add 2
            if t[0] == tmax:
                P[i + 1, j + 1] += 2
            # If shift on second sequence (with gap penalty) yields local max, add 3
            if t[1] == tmax:
                P[i + 1, j + 1] += 3
            # If shift on first sequence (with gap penalty) yields local max, add 4
            if t[2] == tmax:
                P[i + 1, j + 1] += 4  
    
    # Trace through an optimal alignment (starting from last position)
    i, j = nx, ny
    rx, ry = [], []
    while i > 0 or j > 0:
        # If tracer value in [2, 5, 6, 9], it means 2 was added at some point, 
        # hence both sequences were matched -> add them and move back 1 position
        if P[i, j] in [2, 5, 6, 9]:
            rx.append(x[i - 1])
            ry.append(y[j - 1])
            i -= 1
            j -= 1
        # If tracer value in [3, 7], it means 3 was added at some point, 
        # hence shift in second sequence is required -> fill second sequence with gap
        # and only move back 1 position on first sequence
        elif P[i, j] in [3, 7]:
            rx.append(x[i - 1])
            ry.append(fillchar)
            i -= 1
        # Otherwise, tracer value must necessarily be 4, meaning that shift in
        # first sequence is required
        elif P[i, j] in [4]:
            rx.append(fillchar)
            ry.append(y[j - 1])
            j -= 1

    # Reverse outputs
    rx = rx[::-1]
    ry = ry[::-1]

    # Strip both sequences of common gap placeholders at start or end positions
    while rx[0] == fillchar and ry[0] == fillchar:
        rx, ry = rx[1:], ry[1:]
    while rx[-1] == fillchar and ry[-1] == fillchar:
        rx, ry = rx[:-1], ry[:-1]

    # If inputs were strings, convert outputs to strings
    if isinstance(x, str):
        rx = ''.join(rx)
        ry = ''.join(ry)

    # Return outputs tuple
    return (rx, ry)