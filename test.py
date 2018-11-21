#Test for neuron.py
from neuron import Neuron
from neuron import Experiment
from neuron import generate_rhythm
from neuron import print_rhythm_ascii
from neuron import create_tonic
from neuron import create_passive
import neuron
import copy

INFINITY = float('inf')
MINUS_INFINITY = -float('inf')

def create_test_hco():
    n_tonic_pir = Neuron(
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

def test_neuron():
    n_list = create_test_hco()
    print("Tonic with PIR:")
    inhibited_conc = {'ACH':1, 'GLU':0.5}
    print(n_list[0].get_activity(inhibited_conc))
    print(n_list[0].next_state(inhibited_conc))
    zero_conc = {'ACH':0, 'GLU':0}
    print(n_list[0].get_activity(zero_conc))
    print(n_list[0].next_state(zero_conc))


def create_incorrect_tonic_passive():
    #receptors
    receptors_tonic = {'ACH':-1, 'GLU':0}
    receptors_passive = {'ACH':0, 'GLU':1}
    #emission
    emission_tonic = {'ACH':0, 'GLU':1}
    emission_passive = {'ACH':1, 'GLU':0}
    #thresholds
    thresholds_tonic = {-1:(MINUS_INFINITY, -0.5), 0:(-0.5, INFINITY)}
    thresholds_passive = {0:(MINUS_INFINITY, 0.5), 1:(0.5, INFINITY)}
    activity_levels = (0, 1)
    n_tonic = create_tonic('N1', receptors_tonic, emission_tonic, activity_levels,  thresholds_tonic)
    n_passive = create_passive('N2', receptors_passive, emission_passive, activity_levels,  thresholds_passive)
    return [n_tonic, n_passive]

def create_tonic_delayed_passive():
    #receptors
    receptors_tonic = {'ACH':-1, 'GLU':0}
    receptors_passive = {'ACH':0, 'GLU':1}
    #emission
    emission_tonic = {'ACH':0, 'GLU':1}
    emission_passive = {'ACH':1, 'GLU':0}
    #thresholds
    thresholds_tonic = {-1:(MINUS_INFINITY, -0.5), 0:(-0.5, INFINITY)}
    thresholds_passive = {0:(MINUS_INFINITY, 0.5), 1:(0.5, INFINITY)}
    activity_levels = (0, 1)
    n_tonic = create_tonic('N1', receptors_tonic, emission_tonic, activity_levels,  thresholds_tonic)
    n_passive = neuron.create_delayed_passive('N2', receptors_passive, emission_passive, activity_levels,  thresholds_passive)
    return [n_tonic, n_passive]

def test_loop_concurrent_activation():
    neurons = create_incorrect_tonic_passive()
    exp = Experiment("Test experiment", 10, neurons, ['ACH', 'GLU'])
    hist = generate_rhythm(exp)
    assert(not hist[-1].is_valid)
    assert(len(hist) == 2)

def test_tonic_delayed_passive():
    neurons = create_tonic_delayed_passive()
    exp = Experiment("Test experiment", 10, neurons, ['ACH', 'GLU'])
    hist = generate_rhythm(exp)
    assert(hist[-1].is_valid)
    assert(len(hist) == exp.duration+1)
    print("Tonic + delayed passive:")
    print_rhythm_ascii(hist)

def test_experiment():
    neurons = create_test_hco()
    neurons[1].activity_levels = (0, 0.9, 2)
    neurons[0].activity_levels = (0, 0.91, 2)
    exp = Experiment("Test experiment", 10, neurons, ['ACH', 'GLU'])
    hist = generate_rhythm(exp)
    print("Half-center oscillator:")
    print_rhythm_ascii(hist)

def test_oscillator():
    receptors = {'ACH':-1, 'GLU':0}
    #emission
    emission = {'ACH':0, 'GLU':1}
    #thresholds
    thresholds = {-1:(MINUS_INFINITY, -0.5), 0:(-0.5, 0.5), 1:(0.5, INFINITY)}    
    activity_levels = (0, 1)
    burst_duration = 2
    recharge_duration = 3
    n = neuron.create_oscillator("N1", receptors, emission, activity_levels, thresholds, burst_duration, recharge_duration)
    assert(len(n.states) == burst_duration+recharge_duration)
    exp = Experiment("Test experiment", 10, [n], ['ACH', 'GLU'])
    hist = generate_rhythm(exp)
    print("Oscillator:")
    print_rhythm_ascii(hist)



test_neuron()
test_experiment()
test_loop_concurrent_activation()
test_tonic_delayed_passive()
test_oscillator()


