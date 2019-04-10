import neuron as nrn
import copy
import networkx as net
import plot as plt
from plot import plot_history
from graphviz import Digraph
import decimal

INFINITY = float('inf')
MINUS_INFINITY = -float('inf')

def create_hco():
    n_tonic_pir = nrn.Neuron(
    name = 'N1', 
    receptor_weights = { 
        'ACH':-1, 
        'GLU':0}, 
    transmitter_emission = {
        'ACH':0, 
        'GLU':1
    }, 
    activity_levels = (0, 1, 2),
    thresholds = {
        -2: (MINUS_INFINITY, -1),
        -1: (-1, -0.5), 
         0: (-0.5, 0.5), 
         1: (0.5, INFINITY)
    },
    states = ['act', 'inh'],
    state_trans_matrix = { 
        'act':{-2:'inh', -1:'inh', 0:'act', 1:'act'}, 
        'inh':{-2:'inh', -1:'act', 0:'act', 1:'act'}
    }, 
    output_matrix = {
        'act': {-2:0, -1:0, 0:1, 1:2}, 
        'inh': {-2:0, -1:2, 0:2, 1:2}
    }) 
    n_tonic_pir_second = copy.deepcopy(n_tonic_pir)
    n_tonic_pir_second.name = 'N2'
    n_tonic_pir_second.receptor_weights =  { 'ACH': 0, 'GLU': -1 }
    n_tonic_pir_second.transmitter_emission = { 'ACH': 1, 'GLU': 0 }
    return [n_tonic_pir, n_tonic_pir_second]

def demo_hco():
    neurons = create_hco()
    neurons[1].activity_levels = (0, 0.92, 2)
    neurons[0].activity_levels = (0, 0.91, 2)
    exp = nrn.Experiment("Test experiment", 10, neurons, ['ACH', 'GLU'])
    hist = nrn.generate_rhythm(exp)
    print("Half-center oscillator:")
    nrn.print_rhythm_ascii(hist)
    plot_history(hist)
    return neurons

def create_feeding_snail():
    def create_n1():
        n1 = nrn.Neuron(
        'N1',
        {'ach': 0, 'glu':-1},
        {'ach': 1, 'glu':0}, 
        (0, 1), 
        { #thresholds
            -1: (MINUS_INFINITY, -1),
            0: (-1, 1),
            1: (1, INFINITY)
        }, 
        ['charge-0', 'burst'], 
        {#state_trans_matrix 
            'charge-0':{-1:'charge-0', 0:'burst', 1:'burst'}, 
            #'charge-1':{-1:'charge-1', 0:'burst', 1: 'burst'},
            'burst':{-1:'burst', 0:'charge-0', 1:'burst'}
        },
        {#output_trans_matrix 
            'charge-0':{-1:0, 0:0, 1:0}, 
            #'charge-1':{-1:0, 0:0, 1:0},
            'burst':{-1:0, 0:1, 1:1}
        }
        )
        return n1
    def create_n1_double_burst():
        n1 = nrn.Neuron(
        'N1',
        {'ach': 0, 'glu':-1},
        {'ach': 1, 'glu':0}, 
        (0, 1), 
        { #thresholds
            -1: (MINUS_INFINITY, -1),
            0: (-1, 1),
            1: (1, INFINITY)
        }, 
        ['charge-0', 'burst', 'burst-2'], 
        {#state_trans_matrix 
            'charge-0':{-1:'charge-0', 0:'burst', 1:'burst'}, 
            'burst':{-1:'burst', 0:'burst-2', 1:'burst-2'},
            'burst-2':{-1:'charge-0', 0:'charge-0', 1:'burst-2'}
        },
        {#output_trans_matrix 
            'charge-0':{-1:0, 0:0, 1:0}, 
            #'charge-1':{-1:0, 0:0, 1:0},
            'burst':{-1:0, 0:1, 1:1},
            'burst-2':{-1:0, 0:1, 1:1}
        }
        )
        return n1
    def create_n2():
        n2 = nrn.Neuron(
        'N2',
        {'ach': 1, 'glu':0},
        {'ach': 0, 'glu':5}, 
        (0, 1), 
        { #thresholds
            -1: (MINUS_INFINITY, -1),
            0: (-1, 1),
            1: (1, INFINITY)
        }, 
        ['rest', 'burst'],
        {#state_trans_matrix 
            'rest':{-1:'rest', 0:'rest', 1:'burst'}, 
            'burst':{-1:'rest', 0:'rest', 1:'burst'}
        },
        {#output_trans_matrix 
            'rest':{-1:0, 0:0, 1:0}, 
            'burst':{-1:0, 0:1, 1:1}
        }
        )
        return n2
    
    def create_n3():
        n3 = nrn.Neuron(
            'N3', 
            {'ach': -1, 'glu':-0.25},
            {'ach': 0, 'glu':0.5}, 
            (0, 1, 2),
            { #thresholds
                -2: (MINUS_INFINITY, -1),
                -1: (-1, -0.5), 
                0: (-0.5, 0.5), 
                1: (0.5, INFINITY)
            },
            ['act', 'inh'], #states
            {#state_trans_matrix 
                'act':{-2:'inh', -1:'act', 0:'act', 1:'act'}, 
                'inh':{-2:'inh', -1:'act', 0:'act', 1:'act'}
            }, 
            {#output matrix
                'act': {-2:0, -1:0, 0:1, 1:2}, 
                'inh': {-2:0, -1:2, 0:2, 1:2}
            }
        )
        return n3

    #return [create_n1(), create_n2(), create_n3()]
    return [create_n1_double_burst(), create_n2(), create_n3()]
    

def demo_feeding_cpg():
    duration = 20
    T_INJ_ACH = 6
    T_INJ_GLU = 13
    exp = nrn.Experiment('Feeding CPG', duration, create_feeding_snail(), ['ach', 'glu'])
    inj = [{'ach':0, 'glu':0}]*duration
    #delta = {'ach':0, 'glu':0.15}
    FIXED_INJ_ACH = 1
    FIXED_INJ_GLU = 1
    for i in range(T_INJ_ACH, T_INJ_GLU):
        inj[i] = {'ach':FIXED_INJ_ACH, 'glu':0}
    for i in range(T_INJ_GLU, duration):
        inj[i] = {'ach':0, 'glu':FIXED_INJ_GLU}

    exp.injection = inj
    hist = nrn.generate_rhythm_recursive(exp)
    plot_history(hist)

#def print_automaton(neuron):

def print_branch(branch):
    '''
    Branch is a tuple ('order', 'state', 'activity_history')
    '''
    #print( branch.order )
    def format_num(num):
        width = 4
        precision = 2
        return f"{decimal.Decimal(num):{width}.{precision}}"
    neuron_names = sorted([n_name for n_name in branch.activity_history[0]])
    for n_name in neuron_names:#print neurons
        n_str = n_name + '\t|' + '|'.join([format_num(act[n_name]) for act in branch.activity_history]).replace('|0', '| ')
        print(n_str)
    


def demo_recursive_activation():
    neurons = create_hco()
    start_activity = [nrn.get_activities( neurons, nrn._zeros_dict(nrn._list_transmitters(neurons)))]
    branches = nrn._recursive_activation( neurons, start_activity, order=[], injection=[])
    for b in branches:
        print_branch(b)
    exp = nrn.Experiment("Test experiment", 10, neurons, ['ACH', 'GLU'])
    hist = nrn.generate_rhythm_recursive(exp)
    plot_history(hist)

def demo_configuration_space():
    neurons = create_hco()
    injection = {'ACH':0, 'GLU':0}
    all_states, all_transitions = nrn.make_configuration_space(neurons, injection)
    print(all_states)
    print(all_transitions)
    plt.plot_configuration_space(all_states, all_transitions, 'hco_no_inj.gv', \
        transmitter_colors={'ach':'red', 'glu':'green'})
    neurons = create_feeding_snail()
    injection = {'ach':0, 'glu':0}
    all_states, all_transitions = nrn.make_configuration_space(neurons, injection)
    print(all_states)
    print(all_transitions)
    plt.plot_configuration_space(all_states, all_transitions, 'snail_ach_inj.gv', \
        transmitter_colors={'ach':'red', 'glu':'green'})
    plt.plot_branches([all_transitions[0].branch], 'snail_branches_0.gv')
    plt.plot_branches([all_transitions[1].branch], 'snail_branches_1.gv')
    plt.plot_branches([all_transitions[2].branch], 'snail_branches_2.gv')
    

def demo_graphviz():
    dot = Digraph(name='Configuration space')
    n1 = 'burst\n rest\n free'
    n2 = 'burst-1\n burst\n free'
    n3 = 'charge\n rest\n pir'
    dot.node(n1, shape='box')
    dot.node(n2, shape='box')
    dot.node(n3, shape='box')
    dot.edge(n1, n2, color='red', label='ach:1')
    dot.edge(n2, n1, color='green', label='glu:1')
    dot.edge(n2, n3)
    dot.edge(n3, n1)
    dot.render('test.gv', view=True)

def demo_netorkx():
    pass

if __name__ == "__main__":
    #neurons = demo_hco()
    #print(neurons)
    #demo_feeding_cpg()
    #demo_recursive_activation()
    demo_configuration_space()
    #demo_graphviz()
