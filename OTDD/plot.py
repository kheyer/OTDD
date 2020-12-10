import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import datashader as ds
import datashader.transfer_functions as tf
import xarray as xr
from datashader.bundling import connect_edges

import holoviews as hv
from holoviews.operation.datashader import datashade, directly_connect_edges
import holoviews.operation.datashader as hd

hv.extension('bokeh')


def plot_coupling(coupling, distance_matrix, row_labels, col_labels,
                  row_classes, col_classes, figsize=(10,10), cmap='OrRd'):

    row_labels = np.array(row_classes)[row_labels]
    col_labels = np.array(col_classes)[col_labels]
    
    cmap = cm.get_cmap(cmap)

    plt_im = tf.shade(ds.Canvas(plot_width=1200, plot_height=1200).raster(xr.DataArray(coupling*distance_matrix), 
                                interpolate='linear'), cmap=cmap).data

    fig,ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plt_im, cmap=cmap.reversed())

    y_max = coupling.shape[0]
    x_max = coupling.shape[1]

    y_ticks = []
    row = 0
    for c in row_classes:
        num_class = (row_labels==c).sum()
        y_ticks.append((row+num_class//2)*plt_im.shape[0]/y_max)
        row += num_class
        ax.axhline((row)*plt_im.shape[0]/y_max, c='k', alpha=0.5)

    col = 0
    x_ticks = []
    for c in col_classes:
        num_class = (col_labels==c).sum()
        x_ticks.append((col+num_class//2)*plt_im.shape[1]/x_max)
        col += num_class
        ax.axvline(col*plt_im.shape[1]/x_max, c='k', alpha=0.5)
        

    ax.xaxis.set_ticks(x_ticks)
    ax.yaxis.set_ticks(y_ticks)

    ax.set_xlim(0, plt_im.shape[1])
    ax.set_ylim(plt_im.shape[0], 0)

    ax.set_xticklabels(col_classes, rotation=90)
    ax.set_yticklabels(row_classes)

    ax.tick_params(top=True, bottom=False,
                labeltop=True, labelbottom=False)

    return ax

def plot_class_distances(class_distances, row_classes, col_classes, 
                            cmap='OrRd', figsize=(8,8), text=True):
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(class_distances, cmap=cmap)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Distance', rotation=-90, va="bottom")

    ax.set_xticks(np.arange(class_distances.shape[1]))
    ax.set_yticks(np.arange(class_distances.shape[0]))

    ax.set_xticklabels(col_classes, rotation=90)
    ax.set_yticklabels(row_classes)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(class_distances.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(class_distances.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if text:
        for i in range(class_distances.shape[0]):
            for j in range(class_distances.shape[1]):
                text = ax.text(j, i, f'{class_distances[i, j]:.1f}',
                            ha="center", va="center", color="w")
            
    return ax

def plot_connection_graph_ds(node_df, edge_df):

    x_min = node_df.x.min()*1.01
    x_max = node_df.x.max()*1.01
    y_min = node_df.y.min()*1.01
    y_max = node_df.y.max()*1.01

    canvas = ds.Canvas(x_range=(x_min, x_max), y_range=(y_min, y_max))
    edge_plot = tf.shade(canvas.line(connect_edges(node_df, edge_df), 'x','y', agg=ds.count()), 
                     how='log', cmap=['#000000'], alpha=100)

    if 'cat' in node_df.columns:
        node_df.cat = node_df.cat.map(lambda x: str(x))
        node_df.cat = node_df.cat.astype('category')
        aggregator = ds.count_cat('cat')
    else:
        aggregator = None

    agg = canvas.points(node_df, 'x', 'y', aggregator)
    node_plot = tf.spread(tf.shade(agg))
    return tf.stack(node_plot, edge_plot)

def plot_connection_graph_hv(node_df, edge_df):
        
    points = hv.Points(node_df, kdims=['x', 'y'])

    nodes = hv.Nodes(node_df.reset_index(), kdims=['x', 'y', 'index'])

    graph = hv.Graph((edge_df, nodes))

    layout = (datashade(graph, normalization='log', cmap=['#000000'], alpha=150) *
            nodes.opts(size=5, color='cat', cmap='Category20')).opts(height=600, width=600)
    
    return layout

def plot_connection_graph_mpl(node_df, edge_df):

    canvas = ds.Canvas()
    edge_plot = tf.shade(canvas.line(connect_edges(node_df, edge_df), 'x','y', agg=ds.count()), 
                    how='log', cmap=['#000000'])

    xy = node_df[['x', 'y']].values

    xy = np.array(edge_plot.data.shape)*(xy - xy.min(0)) / (xy.max(0) - xy.min(0))

    fig,ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(edge_plot.data, cmap='binary', alpha=0.5)
    scatter = ax.scatter(xy[:,0], xy[:,1], c=node_df.cat, cmap='tab10')

    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
   
def merge_plot_data(xs, ys, x_idxs, y_idxs, x_labels, y_labels):
    node_df = pd.DataFrame(np.concatenate([xs, ys]), columns=['x', 'y'])
    
    if (x_labels is not None) and (y_labels is not None):
        node_df['cat'] = np.concatenate([x_labels, y_labels])
        
    edge_df = pd.DataFrame(x_idxs, columns=['source'])
    edge_df['target'] = y_idxs + xs.shape[0]
    
    return node_df, edge_df

def plot_from_idxs(xs, ys, x_idxs, y_idxs, x_labels, y_labels, plot_type='ds'):
    node_df, edge_df = merge_plot_data(xs, ys, x_idxs, y_idxs, x_labels, y_labels)
    
    if plot_type == 'ds':
        plot = plot_connection_graph_ds(node_df, edge_df)
    elif plot_type == 'mpl':
        plot = plot_connection_graph_mpl(node_df, edge_df)
    else:
        plot = plot_connection_graph_hv(node_df, edge_df)
    
    return plot
    
def plot_distance_network(xs, ys, x_labels, y_labels, M_dist, plot_type='ds'):
    x_idxs = np.array([i for i in range(M_dist.shape[0])])
    y_idxs = M_dist.argmin(-1)
    
    return plot_from_idxs(xs, ys, x_idxs, y_idxs, x_labels, y_labels, plot_type=plot_type)

def plot_coupling_network(xs, ys, x_labels, y_labels, M_coupling, plot_type='ds'):
    x_idxs = np.array([i for i in range(M_coupling.shape[0])])
    y_idxs = M_coupling.argmax(-1)
    
    return plot_from_idxs(xs, ys, x_idxs, y_idxs, x_labels, y_labels, plot_type=plot_type)

def plot_network_threshold(xs, ys, x_labels, y_labels, M, thr, thr_type='greater', plot_type='ds'):
    
    if thr_type == 'greater':
        x_idxs, y_idxs = (M>thr).nonzero()
    else:
        x_idxs, y_idxs = (M<thr).nonzero()
    
    return plot_from_idxs(xs, ys, x_idxs, y_idxs, x_labels, y_labels, plot_type=plot_type)

def plot_network_k_connections(xs, ys, x_labels, y_labels, M, k, k_type='greater', plot_type='ds'):

    # TODO: can this be done faster?
    pairs = np.stack(np.unravel_index(np.argsort(M.ravel()), M.shape)).T
    
    if k_type=='greater':
        pairs = pairs[-k:]
    else:
        pairs = pairs[:k]
        
    x_idxs = pairs[:,0]
    y_idxs = pairs[:,1]
    
    return plot_from_idxs(xs, ys, x_idxs, y_idxs, x_labels, y_labels, plot_type=plot_type)


