import numpy as np
import matplotlib.pyplot as plt
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