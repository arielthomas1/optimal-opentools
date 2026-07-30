"""

    Set of tools and functions for plotting results of the 3D model.

"""
# https://hatarilabs.com/ih-en/how-to-plot-modflow-head-output-in-paraview-with-python-tutorial
import numpy as np
from pyvtk import *
import math

"""
val_arr = hk_arr
out_dir = os.path.join(temp_dir, '_hk')
"""

#   function to create VTK output for plotting in Paraview
def create_VtkFile(z_arr, x_arr, y_arr, val_arr, out_dir):
    #   create a point list
    VTU_Points = []
    #   check for the first non nan value
    for row in range(val_arr.shape[1] + 1):
        for col in range(val_arr.shape[2] + 1):
            if not math.isnan(z_arr[0, row, col]):
                z_last_val = z_arr[0, row, col]
                break
    #   loop through and assign the vertices of each point
    for lay in range(val_arr.shape[0] + 1):
        for row in range(val_arr.shape[1] + 1):
            for col in range(val_arr.shape[2] + 1):
                if z_arr[lay, row, col] != 0 and not math.isnan(z_arr[lay, row, col]):
                    xyz = [x_arr[col], y_arr[row], z_arr[lay, row, col]]
                    VTU_Points.append(xyz)
                    z_last_val = z_arr[lay, row, col]
                else:
                    z_last_arr = z_arr[0, max(0, row - 3) : min(row + 3, val_arr.shape[1]),
                                 max(0, col - 3) : min(col + 3, val_arr.shape[2])]
                    z_last_arr[z_last_arr == 0] = np.nan
                    if math.isnan(np.nanmean(z_last_arr)):
                        xyz = [x_arr[col], y_arr[row], 0.0]
                        VTU_Points.append(xyz)
                    else:
                        z_last_arr = z_arr[0, max(0, row - 10): min(row + 10, val_arr.shape[1]),
                                     max(0, col - 10): min(col + 10, val_arr.shape[2])]
                        z_last_arr[z_last_arr == 0] = np.nan
                        if math.isnan(np.nanmean(z_last_arr)):
                            xyz = [x_arr[col], y_arr[row], np.nanmean(z_last_arr)]
                            VTU_Points.append(xyz)
                        else:
                            #print(np.nanmean(z_last_arr), lay, row, col)
                            xyz = [x_arr[col], y_arr[row], np.nanmean(z_last_arr)]
                            VTU_Points.append(xyz)

    #empty list to store cell coordinates
    listahexahedrons = []
    maximos = []

    #get the nodes and rows per layer
    nodesxlay, nodesxrow = (val_arr.shape[2]+1) * (val_arr.shape[1]+1), val_arr.shape[2]+1
    listaheadsdef = []
    #definition of cell coordinates
    for lay in range(val_arr.shape[0]):
        for row in range(val_arr.shape[1]):
            for col in range(val_arr.shape[2]):
                if z_arr[lay, row, col] != 0 and not math.isnan(z_arr[lay, row, col]):
                    pt0 = nodesxlay * (lay + 1) + nodesxrow * (row + 1) + col
                    pt1 = nodesxlay * (lay + 1) + nodesxrow * (row + 1) + col + 1
                    pt2 = nodesxlay * (lay + 1) + nodesxrow * row + col + 1
                    pt3 = nodesxlay * (lay + 1) + nodesxrow * row + col
                    pt4 = nodesxlay * lay + nodesxrow * (row + 1) + col
                    pt5 = nodesxlay * lay + nodesxrow * (row + 1) + col + 1
                    pt6 = nodesxlay * lay + nodesxrow * row + col + 1
                    pt7 = nodesxlay * lay + nodesxrow * row + col
                    lista = [pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7]
                    listahexahedrons.append(lista)
                    listaheadsdef.append(val_arr[lay, row, col])

    points = VTU_Points
    vtk = VtkData(UnstructuredGrid(points, hexahedron=listahexahedrons),
                  CellData(Scalars(listaheadsdef)),
                  'Unstructured Grid')
    vtk.tofile(out_dir)

#   function to create VTK output for plotting in Paraview
"""
val_arr = hk_arr
qx = qx_in
qz = qz_in
qy = qy_in
out_dir = os.path.join(temp_dir, '_flow')
"""
def create_VtkFile_vector(z_arr, x_arr, y_arr, val_arr, qx, qz, qy, out_dir):
    #   create a point list
    VTU_Points = []
    #   loop through and assign the vertices of each point
    for lay in range(val_arr.shape[0] + 1):
        for row in range(val_arr.shape[1] + 1):
            for col in range(val_arr.shape[2] + 1):
                if z_arr[lay, row, col] != 0 and not math.isnan(z_arr[lay, row, col]):
                    xyz = [x_arr[col], y_arr[row], z_arr[lay, row, col]]
                    VTU_Points.append(xyz)
                    z_last_val = z_arr[lay, row, col]
                else:
                    z_last_arr = z_arr[0, max(0, row - 3) : min(row + 3, val_arr.shape[1]),
                                 max(0, col - 3) : min(col + 3, val_arr.shape[2])]
                    z_last_arr[z_last_arr == 0] = np.nan
                    if math.isnan(np.nanmean(z_last_arr)):
                        xyz = [x_arr[col], y_arr[row], 0.0]
                        VTU_Points.append(xyz)
                    else:
                        z_last_arr = z_arr[0, max(0, row - 10): min(row + 10, val_arr.shape[1]),
                                     max(0, col - 10): min(col + 10, val_arr.shape[2])]
                        z_last_arr[z_last_arr == 0] = np.nan
                        if math.isnan(np.nanmean(z_last_arr)):
                            xyz = [x_arr[col], y_arr[row], np.nanmean(z_last_arr)]
                            VTU_Points.append(xyz)
                        else:
                            #print(np.nanmean(z_last_arr), lay, row, col)
                            xyz = [x_arr[col], y_arr[row], np.nanmean(z_last_arr)]
                            VTU_Points.append(xyz)

    #empty list to store cell coordinates
    listahexahedrons = []
    maximos = []

    #get the nodes and rows per layer
    nodesxlay, nodesxrow = (val_arr.shape[2]+1) * (val_arr.shape[1]+1), val_arr.shape[2]+1
    listaheadsdef = []
    #definition of cell coordinates
    for lay in range(val_arr.shape[0]):
        for row in range(val_arr.shape[1]):
            for col in range(val_arr.shape[2]):
                if z_arr[lay, row, col] != 0 and not math.isnan(z_arr[lay, row, col]):
                    pt0 = nodesxlay * (lay + 1) + nodesxrow * (row + 1) + col
                    pt1 = nodesxlay * (lay + 1) + nodesxrow * (row + 1) + col + 1
                    pt2 = nodesxlay * (lay + 1) + nodesxrow * row + col + 1
                    pt3 = nodesxlay * (lay + 1) + nodesxrow * row + col
                    pt4 = nodesxlay * lay + nodesxrow * (row + 1) + col
                    pt5 = nodesxlay * lay + nodesxrow * (row + 1) + col + 1
                    pt6 = nodesxlay * lay + nodesxrow * row + col + 1
                    pt7 = nodesxlay * lay + nodesxrow * row + col
                    lista = [pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7]
                    listahexahedrons.append(lista)

                    qx_val = qx[lay, row, col]
                    if math.isnan(qx_val):
                        qx_val = 0.
                    qz_val = qz[lay, row, col]
                    if math.isnan(qz_val):
                        qz_val = 0.
                    qy_val = qy[lay, row, col]
                    if math.isnan(qy_val):
                        qy_val = 0.

                    listaheadsdef.append([qx_val, qy_val, qz_val])

    points = VTU_Points
    vtk = VtkData(UnstructuredGrid(points, hexahedron=listahexahedrons),
                  CellData(Vectors(listaheadsdef)),
                  'Unstructured Grid')
    vtk.tofile(out_dir)

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
import matplotlib.ticker as tick

#   function that will plot 2D array input
def plot_2Darray_HK(arr_in, title, out_dir):
    #   create a figure and subplots
    fig = plt.figure(figsize=(12, 16))
    ax2 = plt.subplot2grid((3, 1), (1, 0))  # array
    ax3 = plt.subplot2grid((3, 1), (2, 0))  # color bar area
    #ax1.set_position([0.35, 0.95, 0.3, 0.025])  # [left, bottom, width, height]
    ax2.set_position([0.1, 0.15, 0.85, 0.8])
    ax3.set_position([0.35, 0.025, 0.3, 0.025])

    #   define the colormap
    cmap = cmx.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # define the bins and normalize
    unique_n = [0.0, 0.001, 0.1, 1.0, 5.0, 10., 20.]  # + list(np.arange(10.0, round(max(unique_nan) / 10) * 10, 10.))
    norm = matplotlib.colors.BoundaryNorm(unique_n, cmap.N)
    # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = unique_n
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar_title = 'Hk (m/d))'

    #   plot array
    plot_arr = arr_in
    plot_arr[plot_arr == 0.] = np.nan
    ax2.set_title(title, fontsize=12, y=1.025)
    im1 = ax2.imshow(plot_arr, aspect='auto', interpolation=None, cmap=cmap, norm = norm)
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)

    #   plot the colorbar
    cbar = plt.colorbar(im1, cax=ax3, spacing='uniform', ticks=bounds, boundaries=bounds, orientation='horizontal')
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    cbar.ax.set_title(cbar_title, fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(out_dir, dpi=300)
    plt.close(fig)

#   function that will plot 2D array salinity output
def plot_2Darray_Salinity(arr_in, title, out_dir):
    #   create a figure and subplots
    fig = plt.figure(figsize=(12, 16))
    ax2 = plt.subplot2grid((3, 1), (1, 0))  # array
    ax3 = plt.subplot2grid((3, 1), (2, 0))  # color bar area
    #ax1.set_position([0.35, 0.95, 0.3, 0.025])  # [left, bottom, width, height]
    ax2.set_position([0.1, 0.15, 0.85, 0.8])
    ax3.set_position([0.35, 0.025, 0.3, 0.025])

    #   define the colormap
    cmap = cmx.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # define the bins and normalize
    unique_n = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0, 10., 15., 20., 35.]  # + list(np.arange(10.0, round(max(unique_nan) / 10) * 10, 10.))
    norm = matplotlib.colors.BoundaryNorm(unique_n, cmap.N)
    # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = unique_n
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar_title = 'Salinity (mg/L)'

    #   plot array
    plot_arr = arr_in
    plot_arr[plot_arr < 0.] = 0.
    plot_arr[plot_arr > 35.] = 35.
    ax2.set_title(title, fontsize=12, y=1.025)
    im1 = ax2.imshow(plot_arr, aspect='auto', interpolation=None, cmap=cmap, norm = norm)
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)

    #   plot the colorbar
    cbar = plt.colorbar(im1, cax=ax3, spacing='uniform', ticks=bounds, boundaries=bounds, orientation='horizontal')
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    cbar.ax.set_title(cbar_title, fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(out_dir, dpi=300)
    plt.close(fig)

#   function that will plot 2D array salinity output
def plot_2Darray_Heads(arr_in, title, out_dir):
    #   create a figure and subplots
    fig = plt.figure(figsize=(12, 16))
    ax2 = plt.subplot2grid((3, 1), (1, 0))  # array
    ax3 = plt.subplot2grid((3, 1), (2, 0))  # color bar area
    #ax1.set_position([0.35, 0.95, 0.3, 0.025])  # [left, bottom, width, height]
    ax2.set_position([0.1, 0.15, 0.85, 0.8])
    ax3.set_position([0.25, 0.025, 0.5, 0.025])

    #   define the colormap
    cmap = cmx.twilight
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # define the bins and normalize
    unique_n = [-1000., -100., -10., -5., 0.0, 5., 10., 25., 50., 100, 500, 1000.]
    norm = matplotlib.colors.BoundaryNorm(unique_n, cmap.N)
    # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = unique_n
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar_title = 'Head elevation (m)'

    #   plot array
    plot_arr = arr_in
    #plot_arr[plot_arr < 0.] = 0.
    #plot_arr[plot_arr > 35.] = 35.
    ax2.set_title(title, fontsize=12, y=1.025)
    im1 = ax2.imshow(plot_arr, aspect='auto', interpolation=None, cmap=cmap, norm = norm)
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)

    #   plot the colorbar
    cbar = plt.colorbar(im1, cax=ax3, spacing='uniform', ticks=bounds, boundaries=bounds, orientation='horizontal')
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    cbar.ax.set_title(cbar_title, fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(out_dir, dpi=300)
    plt.close(fig)

#   function that will plot 2D array salinity output
def plot_2Darray_GHB(ghb_in, ghb_cond_in, ssm_in, title, out_dir_ghb, out_dir_ghb_cond, out_dir_ssm):
    #   create a figure and subplots
    fig = plt.figure(figsize=(12, 16))
    ax2 = plt.subplot2grid((3, 1), (1, 0))  # array
    ax3 = plt.subplot2grid((3, 1), (2, 0))  # color bar area
    #ax1.set_position([0.35, 0.95, 0.3, 0.025])  # [left, bottom, width, height]
    ax2.set_position([0.1, 0.15, 0.85, 0.8])
    ax3.set_position([0.25, 0.025, 0.5, 0.025])

    #   define the colormap
    cmap = cmx.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # define the bins and normalize
    unique_n = [-100., -10., 0.0, 5., 10., 15., 20., 30., 50., 100.]
    norm = matplotlib.colors.BoundaryNorm(unique_n, cmap.N)
    # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = unique_n
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar_title = 'GHB elevation (m)'

    #   plot array
    plot_arr = ghb_in
    ax2.set_title(title, fontsize=12, y=1.025)
    im1 = ax2.imshow(plot_arr, aspect='auto', interpolation=None, cmap=cmap, norm = norm)
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)

    #   plot the colorbar
    cbar = plt.colorbar(im1, cax=ax3, spacing='uniform', ticks=bounds, boundaries=bounds, orientation='horizontal')
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    cbar.ax.set_title(cbar_title, fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(out_dir_ghb, dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 16))
    ax2 = plt.subplot2grid((3, 1), (1, 0))  # array
    ax3 = plt.subplot2grid((3, 1), (2, 0))  # color bar area
    #ax1.set_position([0.35, 0.95, 0.3, 0.025])  # [left, bottom, width, height]
    ax2.set_position([0.1, 0.15, 0.85, 0.8])
    ax3.set_position([0.25, 0.025, 0.5, 0.025])

    #   define the colormap
    cmap = cmx.YlGn
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # define the bins and normalize
    unique_n = [0.0, 5., 10., 25., 50., 100, 1000., 1e04, 1e05, 1e06]
    # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = unique_n
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar_title = 'GHB conductance (m2/m)'

    #   plot array
    plot_arr = ghb_cond_in
    ax2.set_title(title, fontsize=12, y=1.025)
    im1 = ax2.imshow(plot_arr, aspect='auto', interpolation=None, cmap=cmap, norm = norm)
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)

    #   plot the colorbar
    cbar = plt.colorbar(im1, cax=ax3, spacing='uniform', ticks=bounds, boundaries=bounds, orientation='horizontal')
    cbar.ax.set_title(cbar_title, fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_xticklabels([str(i) for i in unique_n])

    plt.savefig(out_dir_ghb_cond, dpi=300)
    plt.close(fig)

    #   create a figure and subplots
    fig = plt.figure(figsize=(12, 16))
    ax2 = plt.subplot2grid((3, 1), (1, 0))  # array
    ax3 = plt.subplot2grid((3, 1), (2, 0))  # color bar area
    #ax1.set_position([0.35, 0.95, 0.3, 0.025])  # [left, bottom, width, height]
    ax2.set_position([0.1, 0.15, 0.85, 0.8])
    ax3.set_position([0.25, 0.025, 0.5, 0.025])

    #   define the colormap
    cmap = cmx.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # define the bins and normalize
    unique_n = [0.0, 0.05, 0.1, 0.5, 1.0, 5.0, 10., 15., 20., 35.]
    norm = matplotlib.colors.BoundaryNorm(unique_n, cmap.N)
    # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = unique_n
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar_title = 'Salinity (mg/L)'

    #   plot array
    plot_arr = ssm_in
    #plot_arr[plot_arr < 0.] = 0.
    #plot_arr[plot_arr > 35.] = 35.
    ax2.set_title(title, fontsize=12, y=1.025)
    im1 = ax2.imshow(plot_arr, aspect='auto', interpolation=None, cmap=cmap, norm = norm)
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)

    #   plot the colorbar
    cbar = plt.colorbar(im1, cax=ax3, spacing='uniform', ticks=bounds, boundaries=bounds, orientation='horizontal')
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    cbar.ax.set_title(cbar_title, fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(out_dir_ssm, dpi=300)
    plt.close(fig)

#   function that will plot 2D array salinity output
def plot_2Darray_DRN(drn_in, drn_cond_in, title, out_dir_drn, out_dir_drn_cond):
    #   create a figure and subplots
    fig = plt.figure(figsize=(12, 16))
    ax2 = plt.subplot2grid((3, 1), (1, 0))  # array
    ax3 = plt.subplot2grid((3, 1), (2, 0))  # color bar area
    #ax1.set_position([0.35, 0.95, 0.3, 0.025])  # [left, bottom, width, height]
    ax2.set_position([0.1, 0.15, 0.85, 0.8])
    ax3.set_position([0.25, 0.025, 0.5, 0.025])

    #   define the colormap
    cmap = cmx.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # define the bins and normalize
    unique_n = [-100., -10., 0.0, 5., 10., 15., 20., 30., 50., 100.]
    norm = matplotlib.colors.BoundaryNorm(unique_n, cmap.N)
    # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = unique_n
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar_title = 'DRN elevation (m)'

    #   plot array
    plot_arr = drn_in
    ax2.set_title(title, fontsize=12, y=1.025)
    im1 = ax2.imshow(plot_arr, aspect='auto', interpolation=None, cmap=cmap, norm = norm)
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)

    #   plot the colorbar
    cbar = plt.colorbar(im1, cax=ax3, spacing='uniform', ticks=bounds, boundaries=bounds, orientation='horizontal')
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    cbar.ax.set_title(cbar_title, fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(out_dir_drn, dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 16))
    ax2 = plt.subplot2grid((3, 1), (1, 0))  # array
    ax3 = plt.subplot2grid((3, 1), (2, 0))  # color bar area
    #ax1.set_position([0.35, 0.95, 0.3, 0.025])  # [left, bottom, width, height]
    ax2.set_position([0.1, 0.15, 0.85, 0.8])
    ax3.set_position([0.25, 0.025, 0.5, 0.025])

    #   define the colormap
    cmap = cmx.YlGn
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # define the bins and normalize
    unique_n = [0.0, 5., 10., 25., 50., 100, 1000., 1e04, 1e05, 1e06]
    # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = unique_n
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cbar_title = 'DRN conductance (m2/m)'

    #   plot array
    plot_arr = drn_cond_in
    ax2.set_title(title, fontsize=12, y=1.025)
    im1 = ax2.imshow(plot_arr, aspect='auto', interpolation=None, cmap=cmap, norm = norm)
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)

    #   plot the colorbar
    cbar = plt.colorbar(im1, cax=ax3, spacing='uniform', ticks=bounds, boundaries=bounds, orientation='horizontal')
    cbar.ax.set_title(cbar_title, fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_xticklabels([str(i) for i in unique_n])

    plt.savefig(out_dir_drn_cond, dpi=300)
    plt.close(fig)


