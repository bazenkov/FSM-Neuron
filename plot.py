import matplotlib.pyplot as plt
import numpy as np

def plot_history(history):
    """ Plots the history in a single figure. Each neuron and each transmitter is plotted on a separate axis.
    Plot starts at history[1] as history[0] is the state of not interacting neurons
    history - list of ModelState objects
    """
    plot_options = create_plot_options(history)
    fig, ax_list = plot_neurons(history, plot_options)
    plot_transmitters(fig, ax_list, history, plot_options)
    plt.show()

def create_plot_options(history):
    options = dict()
    options['num_plots'] = _get_num_plots(history)
    options['transmitter_colors'] = ['r', 'g', 'y', 'm']
    #Position =  namedtuple('Position', ['left', 'right', 'top', 'bottom'])
    #options['position'] = Position(300, 800, )
    return options

def plot_neurons(history, options):
    fig, ax_list = plt.subplots(options['num_plots'], 1, 
                                 gridspec_kw=dict(hspace=0.5))
    times = range(1, len(history) )
    for ax in ax_list:
        ax.set_xticks( times )
    n_color = 'k'
    for i_neuron, n_name in enumerate(get_neurons_names(history)):
        n_act = np.array( [history[i].activities[n_name] for i in times ] )
        x = np.array(times)
        #plt.subplot(1 + i_neuron, 1, i_neuron+1)
        ax_list[i_neuron].bar(x, n_act, color=n_color, hatch='/')
        ax_list[i_neuron].set_ylabel( n_name, rotation='horizontal' )
    return fig, ax_list

def plot_transmitters(fig, ax_list, history, options):
    times = range(1, len(history) )
    tr_colors = options['transmitter_colors']
    for i_tr, tr_name in enumerate(transmitter_names(history)):
        tr_conc = np.array( [history[i].concentrations[tr_name] for i in times ] )
        x = np.array(times)
        i_plot = i_tr + _num_neurons(history)
        ax_list[i_plot].bar(x, tr_conc, color=tr_colors[i_tr])
        ax_list[i_plot].set_ylabel(tr_name, rotation='horizontal')
    
def get_neurons_names(history):
    return history[0].activities.keys()

def transmitter_names(history):
    return history[0].concentrations.keys()

def _num_neurons(history):
    return len(history[0].activities)

def _get_num_plots(history):
    return len(history[0].activities)+len(history[0].concentrations)
