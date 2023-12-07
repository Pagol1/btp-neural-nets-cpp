from matplotlib import pyplot as plt
import numpy as np

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, facecolor='w')
d_fp = '''0.9045
0.9217
0.9294
0.9326
0.9357
0.9379
0.9409
0.9424
0.9439
0.9451
0.9455
0.9462
0.9477
0.9483
0.9481
0.9486
0.9491
0.95
0.9504
0.9507'''.split('\n')
a_fp = np.float_(d_fp)*100
ax.plot(a_fp, label="FP32")

a_fxp12 = np.array([0.098]*20)*100
ax2.plot(a_fxp12, label="FXP12")

d_fxp15 = '''0.8995
0.9141
0.919
0.9216
0.9251
0.9288
0.9317
0.9332
0.9343
0.9349
0.9362
0.9368
0.9379
0.9382
0.9386
0.9388
0.9388
0.9392
0.9394
0.9399'''.split("\n")
a_fxp15 = np.float_(d_fxp15)*100
ax.plot(a_fxp15, label="FXP15")

d_fxp24 = '''0.9072
0.9192
0.9257
0.9313
0.9341
0.9362
0.9384
0.9405
0.9423
0.944
0.9448
0.9451
0.946
0.947
0.9473
0.9481
0.9487
0.9493
0.9493
0.9495'''.split("\n")
a_fxp24 = np.float_(d_fxp24)*100
ax.plot(a_fxp24, label="FXP24")

d_fxp32 = '''0.907
0.9239
0.9306
0.9338
0.9366
0.9389
0.9404
0.9423
0.9436
0.9451
0.946
0.9463
0.9471
0.9478
0.948
0.9485
0.9489
0.949
0.9493
0.9497'''.split("\n")
a_fxp32 = np.float_(d_fxp32)*100
ax.plot(a_fxp32, label="FXP32")

# hide the spines between ax and ax2
ax.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
ax2.set_xticklabels([])
fig.text(0.1, 0.5, 'Accuracy', va='center', rotation='vertical')
ax2.set_xlabel("Epochs")

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax.legend()
ax2.legend()
plt.show()
