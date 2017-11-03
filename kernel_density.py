import numpy as np
import matplotlib.pyplot as plt

def kde(features, bandwidth=4.0):
    kp = features.kp
    detector = features.detector

    xx, yy, zz = kd_feature(kp, bandwidth, metric='manhattan')

    plt.pcolormesh(xx, yy, zz)  # , cmap=plt.cm.gist_heat)
    plt.scatter(x=kp[:, 1], y=kp[:, 0], s=2 ** detector.scales, facecolor='white', alpha=.5)
    plt.axis('off')
    plt.show()

def kd_feature(scatter, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""
    from sklearn.neighbors import KernelDensity

    x = scatter[:, 1]; y = scatter[:, 0]

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
             y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)