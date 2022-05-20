# ws = np.linspace(4, 6, 2)
# qs = np.linspace(.06, .14, 3)

# x = raw_timeseries.loc[pd.IndexSlice[:, :, 1:, :], Label.F].groupby([Label.TRIAL, Label.FRAME]).mean()
# xavg = x.groupby(Label.FRAME).mean()
# xb = np.zeros((qs.size, ws.size, x.size))
# xb2 = np.zeros((qs.size, ws.size, x.size))
# xd = np.zeros((qs.size, ws.size, x.size))
# xd2 = np.zeros((qs.size, ws.size, x.size))
# for i, q in enumerate(qs):
#     for j, w in enumerate(ws):
#         wlen = get_window_size(w, fps)
#         xb[i, j] = apply_rolling_window(x, wlen, func=lambda x: x.quantile(q))
#         xd[i, j] = x - xb[i, j] + np.median(xb[i, j])
#         w2 = int(np.round(wlen / 2))
#         if w2 % 2 == 0:
#             w2 += 1
#         xb2[i, j] = apply_rolling_window(xb[i, j], w2)
#         xd2[i, j] = x - xb2[i, j] + np.median(xb2[i, j])

# t = np.arange(x.size) / fps 
# ntrials = len(x.index.unique(level=Label.TRIAL))
# delimiters = np.arange(ntrials) * NFRAMES_PER_TRIAL + FrameIndex.STIM

# fig, axes = plt.subplots(qs.size, 1, figsize=(10, qs.size * 3))
# for i, (ax, q) in enumerate(zip(axes, qs)):
#     sns.despine(ax=ax)
#     ax.set_title(f'q = {q:.2f}')
#     ax.set_xlabel(Label.TIME)
#     ax.plot(t, x, c='k', label='original')
#     for d in delimiters:
#         ax.axvline(d / fps, c='k', ls='--')
#     for j, w in enumerate(ws):
#         line, = ax.plot(t, xb[i, j], ls='--', label=f'w = {w:.1f}s')
#         ax.plot(t, xb2[i, j], ls=':', color=line.get_color())
#         ax.plot(t, xd[i, j], color=line.get_color())
#         ax.plot(t, xd2[i, j], color=line.get_color(), ls='-.')
#     ax.legend()
#     ax.set_xlim(27, 63)
# fig.tight_layout();

# t = np.arange(xavg.size) / fps
# delimiters = [FrameIndex.STIM]
# fig, axes = plt.subplots(qs.size, 1, figsize=(10, qs.size * 3))
# for i, (ax, q) in enumerate(zip(axes, qs)):
#     sns.despine(ax=ax)
#     ax.set_title(f'q = {q:.2f}')
#     ax.set_xlabel(Label.TIME)
#     ax.axhline(0., c='k', ls='--')
#     ax.plot(t, xavg - xavg[FrameIndex.STIM], c='k', label='original')
#     for d in delimiters:
#         ax.axvline(d / fps, c='k', ls='--')
#     for j, w in enumerate(ws):
#         xbavg = pd.Series(xb[i, j], index=x.index).groupby(Label.FRAME).mean()
#         xb2avg = pd.Series(xb2[i, j], index=x.index).groupby(Label.FRAME).mean()
#         xdavg = pd.Series(xd[i, j], index=x.index).groupby(Label.FRAME).mean()
#         xd2avg = pd.Series(xd2[i, j], index=x.index).groupby(Label.FRAME).mean()
#         line, = ax.plot(t, xdavg - xdavg[FrameIndex.STIM], label=f'w = {w:.1f}s')
#         ax.plot(t, xd2avg - xd2avg[FrameIndex.STIM], color=line.get_color(), ls='--')
#         # ax.plot(t, xbavg - xbavg[FrameIndex.STIM], color=line.get_color(), ls='--', )
#         # ax.plot(t, xb2avg - xb2avg[FrameIndex.STIM], color=line.get_color(), ls=':')
#     ax.legend()
# fig.tight_layout();