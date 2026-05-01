import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from cmcrameri import cm

def make_default_fig(figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def subplots(nrows=1, ncols=1, panel_length=4, padding=1):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*(panel_length + padding), nrows*panel_length))
    return fig, ax

def imshow_with_cbar(
    data, fig=None, ax=None, figsize=(5, 5), cmap=cm.grayC, cmap_type=None, title_str=None, ticks='off', labels=None,
    vmin=None, vmax=None, vcenter=None, deltav=None,
    ):

    if fig == None:
        fig, ax = make_default_fig(figsize=figsize)

    if vmin == None:
        vmin = data.min()
    if vmax == None:
        vmax = data.max()
    
    if vcenter is not None:
        if deltav == None:
            deltav = np.max([np.abs(vmax - vcenter), np.abs(vmin - vcenter)])
        vmin = vcenter - deltav
        vmax = vcenter + deltav

    if cmap_type == 'diverging':
        cmap = cm.vik
    elif cmap_type == 'cyclic':
        cmap = cm.vikO
        vmin = -np.pi
        vmax = np.pi
    elif cmap_type == 'sequential':
        cmap = cm.grayC
    elif cmap_type == 'batlow':
        cmap = cm.batlow
    
    if ticks != 'on':
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('scaled')

    if title_str is not None:
        ax.set_title(title_str)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    fig.colorbar(cax, ax=ax)
    fig.tight_layout()

    return (fig, ax)

def plot_tri_grid_advanced(
    ax,
    x,
    y,
    z,
    n_levels=100,
    draw_mesh=True,
    cmap=cm.lipari,
    vmin=None,
    vmax=None,
    levels=None,
    extend='neither',
    fig=None,
    cbar=True,
    cbar_kwargs=None,
    cbar_label=None,
):

    if cbar_kwargs is None:
        cbar_kwargs = {}

    fig = fig if fig is not None else ax.figure

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # case #1: regular grid axes
    if x.ndim == 1 and y.ndim == 1 and z.ndim == 2:
        if z.shape == (len(y), len(x)):
            X, Y = np.meshgrid(x, y, indexing='xy')
            xx, yy, zz = X.ravel(), Y.ravel(), z.ravel()
        elif z.shape == (len(x), len(y)):
            X, Y = np.meshgrid(x, y, indexing='xy')
            xx, yy, zz = X.ravel(), Y.ravel(), z.T.ravel()
        else:
            raise ValueError(
                f'z has shape {z.shape}, but expected (len(y),len(x))={len(y),len(x)} '
                f'or (len(x),len(y))={len(x),len(y)}'
            )
    
    # case #2: scattered points
    else:
        xx = np.ravel(x)
        yy = np.ravel(y)
        zz = np.ravel(z)
        if xx.shape != yy.shape or xx.shape != zz.shape:
            raise ValueError(
                f'x, y, z must have the same length for scattered input. '
                f'Got {xx.shape}, {yy.shape}, {zz.shape}'
            )

    # enforce color range via levels
    if levels is None:
        if vmin is not None and vmax is not None:
            levels = np.linspace(vmin, vmax, int(n_levels) + 1)
        else:
            levels = int(n_levels)

    triang = mtri.Triangulation(xx, yy)

    cs = ax.tricontourf(
        triang, zz,
        levels=levels,
        cmap=cmap,
        extend=extend,
    )

    if draw_mesh:
        ax.triplot(triang, color='k', linewidth=0.1, alpha=0.4)

    cb = None
    if cbar:
        cb = fig.colorbar(cs, ax=ax, **cbar_kwargs)
        if cbar_label is not None:
            cb.set_label(cbar_label)

    return cs, triang, cb