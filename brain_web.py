from __future__ import print_function, division

import brainweb
import numpy as np
from brainweb import noise, T1, Res, Shape, Act
from skimage.transform import resize


# method copied from brainweb module to modify according to our requirement to visualize as an image
def convert_to_T1(im, pad=True, dtype=np.float32, outres="MR"):
    """
      @param outres: attribute to use from `Res` & `Shape` classes [default: "MR"]
      @return out  : image data as an array
      """
    out_res = getattr(Res, outres)
    out_shape = getattr(Shape, outres)

    new_shape = np.rint(np.asarray(im.shape) * Res.brainweb / out_res)
    padLR, padR = divmod((np.array(out_shape) - new_shape), 2)

    def resizeToMmr(arr):
        arr = resize(arr, new_shape,
                     order=1, mode='constant', anti_aliasing=False,
                     preserve_range=True).astype(dtype)
        if pad:
            arr = np.pad(arr, [(p, p + r) for (p, r)
                               in zip(padLR.astype(int), padR.astype(int))],
                         mode="constant")
        return arr

    arr = np.zeros_like(im, dtype=dtype)  # dtype only needed for uMap?
    for attr in T1.attrs:
        arr[Act.indices(im, attr)] = getattr(T1, attr)
    res = resizeToMmr(arr)

    return res


# method copied from brainweb module to modify according to our requirement to download as an image
def volshow(vol,
            cmaps=None, colorbars=None,
            xlabels=None, ylabels=None, titles=None,
            vmins=None, vmaxs=None,
            sharex=True, sharey=True,
            ncols=None, nrows=None,
            figsize=None, frameon=True, tight_layout=1,
            filename=None, fontproperties=None):
    """
    Interactively slice through 3D array(s) in Jupyter

    @param vol  : imarray or [imarray, ...] or {'title': imarray, ...}
      Note that imarray may be 3D (mono) or 4D (last channel rgb(a))
    @param cmaps  : list of cmap [default: ["Greys_r", ...]]
    @param xlabels, ylabels, titles  : list of strings (default blank)
    @param vmins, vmaxs  : list of numbers [default: [None, ...]]
    @param colorbars  : list of bool [default: [False, ...]]
    @param sharex, sharey, ncols, nrows  : passed to
      `matplotlib.pyplot.subplots`
    @param figsize, frameon  : passed to `matplotlib.pyplot.figure`
    @param tight_layout  : number of times to run `tight_layout(0, 0, 0)`
      [default: 1]
    """
    from IPython.display import display
    import ipywidgets as ipyw
    import matplotlib.pyplot as plt

    if hasattr(vol, "keys") and hasattr(vol, "values"):
        if titles is None:
            titles = vol.keys()
            vol = list(vol.values())

    if vol[0].ndim == 2:  # single 3darray
        vol = [vol]
    else:
        for v in vol:
            if v.ndim not in [3, 4]:
                raise IndexError("Input should be (one or a list of)"
                                 " 3D and/or 4D array(s)")

    if cmaps is None:
        cmaps = ["Greys_r"] * len(vol)
    if colorbars is None:
        colorbars = [False] * len(vol)
    if xlabels is None:
        xlabels = [""] * len(vol)
    if ylabels is None:
        ylabels = [""] * len(vol)
    if titles is None:
        titles = [""] * len(vol)
    if vmins is None:
        vmins = [None] * len(vol)
    if vmaxs is None:
        vmaxs = [None] * len(vol)
    if tight_layout in (True, False):
        tight_layout = 1 if tight_layout else 0

    # automatically square-ish grid, slightly favouring more rows
    if nrows:
        rows = nrows
        cols = ncols or int(np.ceil(len(vol) / rows))
    else:
        cols = ncols or max(1, int(len(vol) ** 0.5))
        rows = int(np.ceil(len(vol) / cols))
    # special case
    if not (nrows or ncols) and len(vol) == 4:
        nrows = ncols = 2

    zSize = min(len(i) for i in vol)

    # matplotlib>=3.3.2 figure updating needs its own output area
    # https://github.com/matplotlib/matplotlib/issues/18638
    out = ipyw.Output()
    display(out)
    with out:
        fig = plt.figure(figsize=figsize, frameon=frameon)

    @ipyw.interact(z=ipyw.IntSlider(zSize // 2, 0, zSize - 1, 1))
    def plot_slice(z):
        """z  : int, slice index"""
        plt.figure(fig.number, clear=True)
        axs = fig.subplots(rows, cols, sharex=sharex, sharey=sharey)
        axs = list(getattr(axs, 'flat', [axs]))
        for ax, v, cmap, cbar, xlab, ylab, tit, vmin, vmax in zip(
                axs, vol, cmaps, colorbars,
                xlabels, ylabels, titles,
                vmins, vmaxs):
            plt.sca(ax)
            plt.imshow(v[z], cmap=cmap, vmin=vmin, vmax=vmax)
            if cbar:
                plt.colorbar()
            textargs = {}
            if fontproperties is not None:
                textargs.update(fontproperties=fontproperties)
            if xlab:
                plt.xlabel(xlab, **textargs)
            if ylab:
                plt.ylabel(ylab, **textargs)
            if tit:
                plt.title(tit, **textargs)

            ax.set_xticks([])
            ax.set_yticks([])
            if filename is not None:
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.show()
            if not frameon:
                plt.setp(ax.spines.values(), color='white')
                #  don't need all axes if sharex=sharey=True
                ax.set_xticks(())
                ax.set_yticks(())

        for _ in range(tight_layout):
            plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        # make sure to clear extra axes
        for ax in axs[axs.index(ax) + 1:]:
            ax.axis('off')
        # return fig, axs

    return plot_slice


# source dataset files were downloaded using brainweb module
# source dataset file used to generate image for testing
file = 'subjects/subject_20.bin.gz'

# load the raw image data
raw_data = brainweb.load_file(file)

# convert raw data to T1 image data
t1 = convert_to_T1(raw_data, pad=False, outres="MR")

# show and download T1 original image
volshow(t1, cmaps=['Greys_r'], filename='images/t1_original_20.png')
