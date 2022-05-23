# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2022-05-23 16:54:33
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2022-05-23 17:03:25

import numpy as np
import matplotlib.pyplot as plt
from postpro import pvalue_to_tscore, pvalue_to_zscore

n = np.arange(20) + 1
probs = [.01, .05, .1]

fig, ax = plt.subplots()
ax.set_xlabel('sample size')
ax.set_ylabel('threshold t-score')
ax.axvline(10, ls='--', c='k')
ax.set_ylim(0, 5)
for p in probs:
    t = np.array([pvalue_to_tscore(p, nn) for nn in n])
    z = pvalue_to_zscore(p)
    line, = ax.plot(n, t / z, label=f'p = {p}')
    # ax.plot(n, t / np.sqrt(n), ls='--', c=line.get_color())
    # ax.axhline(z, ls=':', c=line.get_color())
    # ax.plot(n, np.ones_like(n) * z, ls=':', c=line.get_color())
ax.legend()
plt.show()