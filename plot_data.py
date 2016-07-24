import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

sns.set_style('white')
pretty_labels = {
    's_pos': r'$\phi$',
    's_avg': r'$\langle s \rangle$',
    'x_avg': r'$\langle x \rangle$',
    'gdp_avg': 'GDP',
    'utility_avg': r'$u$',
    'spread_reduction': r'$\xi \cdot \delta x_0$'
}


def plot_array(data_name):
    arr = data[data_name]
    fig, ax = pl.subplots()
    ax.plot(n_list, np.average(arr, axis=1), 'o')
    ax.axvline(x=2, color='k', linestyle='--', linewidth=1.)
    ax.set_xlabel('$n$', fontsize=24)
    ax.set_ylabel(pretty_labels[data_name], fontsize=24)
    sns.despine()
    fig.savefig(data_name + '.png', bbox_to_inches='tight')

data = np.load('data.npz')
n_list = data['n']

for data_name in data.files:
    if data_name != 'n':
        plot_array(data_name)
