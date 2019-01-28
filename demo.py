import neuron as nrn
import copy
#import networkx as net
from plot import plot_history

INFINITY = float('inf')
MINUS_INFINITY = -float('inf')

def create_hco():
    n_tonic_pir = nrn.Neuron(
    'N1', #name
    {#receptor_weights 
        'ACH':-1, 
        'GLU':0}, 
    {#emission
        'ACH':0, 
        'GLU':1
    }, 
    (0, 1, 2), #activity levels
    { #thresholds
        -2: (MINUS_INFINITY, -1),
        -1: (-1, -0.5), 
         0: (-0.5, 0.5), 
         1: (0.5, INFINITY)
    },
    ['act', 'inh'], #states
    {#state_trans_matrix 
        'act':{-2:'inh', -1:'inh', 0:'act', 1:'act'}, 
        'inh':{-2:'inh', -1:'act', 0:'act', 1:'act'}
    }, 
    {#output matrix
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
            'N3t', 
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
    duration = 15
    exp = nrn.Experiment('Feeding CPG', duration, create_feeding_snail(), ['ach', 'glu'])
    hist = nrn.generate_rhythm(exp)
   # plot_history(hist)
    inj = [{'ach':0, 'glu':0}]*duration
    for i in range(duration // 2,duration):
        inj[i] = {'ach':inj[i-1]['ach']+0.2, 'glu':0}
    exp.injection = inj
    hist = nrn.generate_rhythm(exp)
    plot_history(hist)

#def print_automaton(neuron):

if __name__ == "__main__":
    #neurons = demo_hco()
    #print(neurons)
    demo_feeding_cpg()

