"""Module providing functions to flexibly plot maps using matplotlib and cartopy."""

import cartopy.crs as ccrs
import geopandas
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.io import shapereader
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Letters for plot labels.
letters = [
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'i',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'w',
    'x',
    'y',
    'z',
]

def plot_map_to_ax(
    dat,
    ax,
    coastlines=True,
    grid=True,
    dat_proj=ccrs.PlateCarree(),
    disp_proj=ccrs.PlateCarree(),
    title=None,
    colour_scale=None,
    cmap=None,
    norm=None,
    cbar_ticks=None,
    tick_labels=None,
    contour=False,
    stippling=None,
    stipple_size=3,
    colourbar=True,
    ticks_left=True,
    ticks_bottom=True,
    cbar_aspect=25,
    cbar_fraction=0.07,
    cbar_shrink=0.4,
    cbar_pad=0.015,
    cbar_label=None,
    cbar_orientation='vertical',
    coastlines_colour='black',
    xlims=None,
    ylims=None,
    num_ticks=None,
    divergent=False,
    cbar_inset=False,
    title_inset=False,
    discrete=False,
    log_scale=False,
    nan_colour='#eeeeee',
    axis_off=False,
    country=None,
    annotations=None,
    polygons=None,
    polygon_colour='black',
    hatch='.',
    pts=None,
    left_title=None,
):
    """Plot data on a map to a specified plot axis object.

    Arguments:
        dat: DataSet to plot or list of datasets to plot.
        ax: GeoAxes object to plot to.
        dat_proj, dist_proj: Data and display projections.
        figsize: Figure size width x height.
        coastlines: Show coastlines?
        grid: Show grid?
        ncol/nrows: Number of columns/rows to plot.
        title: Title for the plot.
        colour_scale: None for default, or a tuple of min/max values for the scale.
        cmap: The matplotlib colour map to use.
        norm: A norm object for colours (e.g. colors.BoundaryNorm).
        cbar_ticks: Colour bar ticks.
        tick_labels: Colour bar tick labels.
        contour: Plot using xarray's contourf function?
        stippling: True where trippling should appear.
        stipple_size: Size for stippling points.
        colourbar: Include a colourbar?
        ticks_left: Include ticks on left of plot?
        ticks_bottom: Include ticks on bottom of plot?
        cbar_aspect: colorbar aspect ratio?
        cbar_fraction: fraction argument for colorbar().
        cbar_shrink: shrink argument for colorbar().
        cbar_pad: pad argument for colorbar().
        cbar_label: Overwrite label?
        cbar_orientation: orientation argument for colorbar().
        coastlines_colour: Colour for coastlines.
        xlims: x plot limit.
        ylims: y plot limits.
        num_ticks: Number of ticks for x and y axes (None for auto).
        divergent: Is the colour scale divergent? If so make zero central.
        cbar_inset: Inset the colorbar in lower left?
        title_inset: Inset the title in the upper left?
        discrete: Make the colour bar discrete?
        log_scale: Make the colour scale log-scaled?
        nan_colour: Colour for missing values.
        axis_off: Turn off all axes.
        country: Plot coastlines only for a specific country.
        annotations: Add annotations to the map - dictionary of {'Text': [x, y, xadj, yadj, ha]} where
                     x, y are position of text, xadj, yadj give offsets to the label, ha is 'left' or
                     'right' for horizontal anchor.
        polygons: If specified, draw each polygon onto the plot.
        polygon_colour: Colour for polygons.
        hatch: Hatching to use for stippling ('.' for scatterplot, '//' etc for contour hatching).
        dat_proj: The data projection.
        disp_proj: Projection to plot in.
        left_title: Put titles to left instead of centre?
        pts: Extra points to plot.

    """
    col_min = None
    col_max = None

    # Rasterize elements with zorder below 0.
    ax.set_rasterization_zorder(0)

    if colour_scale is not None:
        col_min = colour_scale[0]
        col_max = colour_scale[1]

    if divergent:
        if col_min is None or col_max is None:
            col_min = dat.min()
            col_max = dat.max()

        col_min = -1 * np.max(np.abs([col_min, col_max]))
        col_max = np.max(np.abs([col_min, col_max]))

    cbar_spacing = 'proportional'

    if discrete:
        assert cbar_ticks is not None, 'Discrete colorbar requires cbar_ticks'
        assert cbar_ticks == sorted(cbar_ticks), 'cbar_ticks must be sorted for discrete plot.'
        cbar_ticks = np.array([*cbar_ticks, np.max(cbar_ticks) + 1])
        norm = colors.BoundaryNorm(cbar_ticks, ncolors=len(cbar_ticks) - 1)
        cbar_ticks = (cbar_ticks[0:-1] + cbar_ticks[1:]) / 2
        cbar_spacing = 'uniform'

    if log_scale:
        norm = colors.LogNorm(vmin=col_min, vmax=col_max)

    fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    fmt.set_powerlimits((-4, 6))
    cbar_args = {
        'spacing': cbar_spacing,
        'fraction': cbar_fraction,
        'ticks': cbar_ticks,
        'aspect': cbar_aspect,
        'shrink': cbar_shrink,
        'pad': cbar_pad,
        'orientation': cbar_orientation,
        'format': '%g',
    }

    if cbar_inset:
        cax = inset_axes(
            ax,
            width='50%',
            height='3%',
            loc='lower left',
            bbox_to_anchor=(0.05, 0.15, 1, 1),
            bbox_transform=ax.transAxes,
        )
        cbar_args['cax'] = cax

    if cbar_label is not None:
        cbar_args['label'] = cbar_label

    if colourbar is False:
        cbar_args = None

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(nan_colour)

    if not contour:
        res = dat.plot(
            ax=ax,
            transform=dat_proj,
            vmin=col_min,
            vmax=col_max,
            cmap=cmap,
            norm=norm,
            cbar_kwargs=cbar_args,
            add_colorbar=colourbar,
            zorder=-10,
        )
    else:
        res = dat.plot.contourf(
            ax=ax,
            transform=dat_proj,
            vmin=col_min,
            vmax=col_max,
            cmap=cmap,
            norm=norm,
            cbar_kwargs=cbar_args,
            add_colorbar=colourbar,
        )

    if stippling is not None:
        ax.autoscale(False)  # noqa: FBT003
        if hatch == '.':
            pts = stippling.where(stippling).to_dataframe().dropna().reset_index()
            ax.scatter(
                x=pts[stippling.dims[1]],
                y=pts[stippling.dims[0]],
                marker='.',
                color='black',
                transform=dat_proj,
                s=stipple_size,
            )
        else:
            stippling.plot.contourf(hatches=['', hatch], levels=[0, 0.5, 1], colors='none', ax=ax, add_colorbar=False)

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    if tick_labels is not None:
        assert len(tick_labels) == len(cbar_ticks), 'Labels and ticks must have same length'
        res.colorbar.ax.set_yticklabels(tick_labels)
    if left_title is not None:
        if title_inset:
            title = f'{left_title} {title}'
        else:
            ax.set_title(left_title, fontsize=plt.rcParams['font.size'], loc='left')
    if title is not None:
        if title_inset:
            ax.annotate(
                text=title,
                xy=(0.05, 0.9),
                xycoords='axes fraction',
                fontweight='bold',
                fontsize=plt.rcParams['font.size'],
            )
        else:
            ax.set_title(title, fontsize=plt.rcParams['font.size'])
    if polygons is not None:
        poly = geopandas.GeoSeries(polygons).unary_union
        ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor='none', edgecolor=polygon_colour, linewidth=1.75)
    if coastlines:
        if country is not None:
            shpfilename = shapereader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
            df = geopandas.read_file(shpfilename)
            poly = df.loc[df['ADMIN'] == country]['geometry'].values[0]
            ax.add_geometries(
                poly,
                crs=ccrs.PlateCarree(),
                facecolor='none',
                edgecolor=coastlines_colour,
                linewidth=0.75,
            )
        else:
            ax.coastlines(color=coastlines_colour)
    if grid:
        locator = None
        if num_ticks is not None:
            locator = mticker.MaxNLocator(nbins=num_ticks + 1)
        gl = ax.gridlines(crs=disp_proj, draw_labels=True, alpha=0.5, xlocs=locator, ylocs=locator)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = ticks_left
        gl.bottom_labels = ticks_bottom
    if axis_off:
        ax.axis('off')

    if pts is not None:
        for i in pts:
            ax.scatter(pts[i][0], pts[i][1], color='black')

    if annotations is not None:
        for text, [x, y, xadj, yadj, ha] in annotations.items():
            if np.abs(xadj) >= 1 or np.abs(yadj) >= 1:
                if ha in ('right', 'left'):
                    ax.plot([x, x + xadj - (0.2 * np.sign(xadj))], [y, y + yadj + 0.2], color='black')
                elif ha == 'center':
                    ax.plot([x, x + xadj - (0.2 * np.sign(xadj))], [y, y + yadj - 0.2], color='black')
                    if yadj < 0:
                        print('Warning: ha=center and negative y adjustment are not supported.')
                else:
                    assert 1 == 0, 'Invalid value of ha.'  # noqa: PLR0133
            ax.annotate(xy=(x + xadj, y + yadj), text=text, ha=ha)

    return res


def plot_map(
    dat,
    dat_proj=ccrs.PlateCarree(),
    disp_proj=ccrs.PlateCarree(),
    figsize=(12, 8),
    grid=True,
    ncols=1,
    nrows=1,
    title=None,
    share_scale=False,
    colour_scale=None,
    cbar_ticks=None,
    tick_labels=None,
    file=None,
    scale_label='',
    share_axes=False,
    ticks_left=True,
    ticks_bottom=True,
    wspace=0.05,
    hspace=0.05,
    stippling=None,
    cbar_adjust=0.862,
    cbar_pad=0.015,
    col_labels=None,
    row_labels=None,
    xlims=None,
    ylims=None,
    show=True,
    shared_scale_quantiles=(0, 1),
    polygons=None,
    letter_labels=False,
    **kwargs,
):
    """Plot data on a map.

    Arguments:
        dat: DataSet to plot or list of datasets to plot.
        dat_proj, dist_proj: Data and display projections.
        figsize: Figure size width x height.
        grid: Show grid?
        ncols/nrows: Number of columns/rows to plot.
        title: Title(s) for the plot(s).
        share_scale: Make the range of values in each plot the same?
        colour_scale: Tuple with min/max values to use on scale. Overwritten by share_scale.
        cbar_ticks: Ticks for the colourbar.
        tick_labels: Colour bar tick labels.
        file: If specified save to 'file' instead of showing onscreen.
        scale_label: The label for a shared scale.
        share_axes: Share left/bottom axes?
        ticks_left, ticks_bottom: Display ticks on the left/bottom of plots?
        wspace, hspace: gridspec wspace and hspace arguments.
        stippling: Stippling per axis.
        cbar_adjust: Amount to shrink plots by to add cbar for shared scale.
        cbar_pad: Padding between figure and colour bar.
        col_labels/row_labels: Labels for each column/row; overwrites individial plot titles.
        xlims: x limits.
        ylims: y limits.
        show: Show the map?
        shared_scale_quantiles: Quantiles for a shared scale.
        polygon: If specified, polygons to put on each map.
        letter_labels: Use a letter to label each subplot?
        kwargs: Extra arguments to plot_map_to_ax.
        col_labels: Labels for each column.
        dat_proj: Data projection.
        disp_proj: Display projection.
        hspace: Height spacing.
        wspace: Width spacing.
        ncols: Number of columns to use.
        nrows: Number of rows.
        polygons: Polygons to overplot.
        row_labels: Labels for each row.
        ticks_bottom: Include bottom ticks?
        ticks_left: Include left ticks?

    Return:
        - The axis plotted to.

    """
    fig, ax = plt.subplots(
        figsize=figsize,
        ncols=ncols,
        nrows=nrows,
        subplot_kw={'projection': disp_proj},
        gridspec_kw={'wspace': wspace, 'hspace': hspace},
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if not isinstance(dat, list):
        im = plot_map_to_ax(
            dat=dat,
            ax=ax,
            grid=grid,
            dat_proj=dat_proj,
            disp_proj=disp_proj,
            title=title,
            stippling=stippling,
            colour_scale=colour_scale,
            cbar_ticks=cbar_ticks,
            cbar_pad=cbar_pad,
            tick_labels=tick_labels,
            ticks_left=ticks_left,
            ticks_bottom=ticks_bottom,
            xlims=xlims,
            ylims=ylims,
            polygons=polygons,
            **kwargs,
        )
    else:
        assert ncols * nrows >= len(dat), 'Not enough cols/rows to fit all plots.'

        if share_scale:
            all_vals = np.array([])

            for d in dat:
                all_vals = np.concatenate([all_vals, np.array(d.values.flat)])

            colour_scale = (
                np.nanquantile(all_vals, shared_scale_quantiles[0]),
                np.nanquantile(all_vals, shared_scale_quantiles[1]),
            )
            assert not (np.isnan(colour_scale[0]) or np.isnan(colour_scale[1])), 'share_scale cannot be used with subplots missing data.'

        for i, d in enumerate(dat):
            ax_title = None
            if title is not None:
                ax_title = title[i]

            ax_poly = None
            if polygons is not None:
                ax_poly = polygons[i]

            tb = ticks_bottom
            tl = ticks_left
            if share_axes:
                if i < (ncols * nrows) - ncols:
                    tb = False
                if i % ncols != 0:
                    tl = False

            stipple = None if stippling is None else stippling[i]
            proj = dat_proj if not isinstance(dat_proj, list) else dat_proj[i]
            xlim = xlims if not isinstance(xlims, list) else xlims[i]
            ylim = ylims if not isinstance(ylims, list) else ylims[i]

            left_title = None
            if letter_labels:
                left_title = letters.pop(0)

            im = plot_map_to_ax(
                dat=d,
                ax=ax.flat[i],
                grid=grid,
                dat_proj=proj,
                disp_proj=disp_proj,
                title=ax_title,
                colour_scale=colour_scale,
                cbar_pad=cbar_pad,
                cbar_ticks=cbar_ticks,
                tick_labels=tick_labels,
                colourbar=(not share_scale),
                stippling=stipple,
                xlims=xlim,
                ylims=ylim,
                ticks_left=tl,
                ticks_bottom=tb,
                polygons=ax_poly,
                left_title=left_title,
                **kwargs,
            )

        while i + 1 < len(ax.flat):
            fig.delaxes(ax.flat[i + 1])
            i = i + 1

        if share_scale:
            fig.subplots_adjust(right=cbar_adjust)
            cbar_ax = fig.add_axes([cbar_adjust + cbar_pad, 0.23, 0.02, 0.55])
            fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
            fmt.set_powerlimits((-4, 6))
            _ = fig.colorbar(im, ax=ax, cax=cbar_ax, ticks=cbar_ticks, label=scale_label, format=fmt)

        if col_labels is not None or row_labels is not None:
            for a in ax.flat:
                a.set_title('')

            if col_labels is not None:
                axes = ax if ax.ndim == 1 else ax[0, :]
                for a, lab in zip(axes, col_labels):
                    a.set_title(lab, fontsize=plt.rcParams['font.size'])

            if row_labels is not None:
                fig.subplots_adjust(left=0.02)
                lab_ax = fig.add_axes([0, 0.11, 0.02, 0.78], autoscale_on=True)
                lab_ax.axis('off')

                for i, lab in enumerate(row_labels):
                    p = 0.03 + (i / len(row_labels)) * 1.33
                    lab_ax.annotate(lab, xy=(0.5, 1 - p), rotation=90, xycoords='axes fraction', ha='center')

    if file is not None:
        plt.savefig(fname=file, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()
    elif show:
        plt.show()

    return ax
