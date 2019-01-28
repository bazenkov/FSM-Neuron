""" Neuron as a finite state machine """
from random import choice
from collections import Counter
from collections import OrderedDict
from collections import namedtuple
import copy
import decimal

class Neuron:

    def __init__(self, name, receptor_weights,
        transmitter_emission, activity_levels, thresholds,
        states, state_trans_matrix, output_matrix):
        """ Properties:
            - name - the unique name
            - receptor_weights {'ACH':(-1, -0.5), 'GLU':(0, 0)}
            - transmitter_emission {'ACH':0, 'GLU':1}
            - activity_levels (0, 1, 2)
            - thresholds - dict { -1:(-inf, -0.5), 0:(-0.5, 0.5), 1:(0.5, inf)} 
            - states ('s0', 's1')
            - state_trans_matrix {'s0':{-1:'s1', 0:'s0', 1:'s0'}, 's1':{-1:'s1', 0:'s0', 1:'s0'}} 
            - output_matrix {'s0':{-1:0, 0:1, 1:2}, 's1':{-1:0, 0:0, 1:0}}
            - current state
        """
        self.name = name
        self.receptor_weights = copy.copy(receptor_weights)
        self.max_delay = 0
        self.transmitter_emission = copy.copy(transmitter_emission)
        self.activity_levels = copy.copy(activity_levels)
        self.thresholds = Neuron._make_thresholds(copy.copy(thresholds))
        self.states = copy.copy(states)
        self.state_trans_matrix = copy.copy(state_trans_matrix)
        self.output_matrix = copy.copy(output_matrix)
        self.current_state = states[0]
        self._check_consistency()

    Range = namedtuple('Range', ['low', 'up'])

    @staticmethod
    def _make_thresholds(thresholds):
        named = [(t[0], Neuron.Range(t[1][0], t[1][1])) for t in thresholds.items()]
        return OrderedDict( sorted(named, key=lambda t: t[0] ))

    def transmitter_names(self):
        return list(self.transmitter_emission.keys())
        
    def get_total_input(self, concentrations):
        """ concentrations - dict {'ACH':0, 'GLU':0.5} """
        s = 0
        #print(self.receptor_weights)
        for trans in concentrations:
            s += concentrations[trans]*self.receptor_weights[trans]
        return s

    def next_state(self, concentrations):    
        discretized = self.discrete_input(self.get_total_input(concentrations))
        self.current_state = self.state_trans_matrix[self.current_state][discretized]
        return self.get_current_state()
    
    def get_activity(self, concentrations):
        """ get next activity for the concentrations """
        #print(concentrations)
        total_input = self.get_total_input(concentrations)
        discretized = self.discrete_input(total_input)
        act_level = self.output_matrix[self.get_current_state()][discretized]
        return self.activity_levels[act_level]

    def discrete_input(self, total_input):
        """Get discretized input"""    
        for tr in self.thresholds.keys():
            if total_input >= self.thresholds[tr].low and total_input < self.thresholds[tr].up:
                return tr
        raise ValueError("Incorrect discretization")

    def get_current_state(self):
        return self.current_state
    
    def get_time(self, concentrations, past_concentrations):
        """ Returns the time used to sort neurons' firing order """
        delta = self.get_total_input(concentrations)
        past_delta = self.get_total_input(past_concentrations)
        d_inp = self.discrete_input(delta) 
        up_b = self.thresholds[d_inp].up
        low_b = self.thresholds[d_inp].low
        if delta == 0:
            return float("inf")
        if past_delta > up_b:
            return (past_delta - up_b)/abs(delta)
        elif past_delta <= low_b:
            return (low_b - past_delta)/abs(delta)
        else:
            return float("inf")
        # up_bound = self.thresholds
        #return 1/(1+self.get_activity(concentrations))

    def _check_consistency(self):
        self._check_transmitter_names()
        Neuron._check_thresholds(self.thresholds)
        self._check_automaton()
    
    def _check_transmitter_names(self):
        inp_trans = self.receptor_weights.keys()
        out_trans = self.transmitter_emission.keys()
        if Counter(inp_trans) != Counter(out_trans):
            raise ValueError("Neuron " + self.name + " has different input and output transmitters:\n" +
                            "Input: " + str(inp_trans) + "\n" + 
                            "Output: " + str(out_trans))
    
    @staticmethod
    def _check_thresholds(thresholds):
        #starts with -inf, ends with +inf
        if thresholds[min(thresholds.keys())].low != -float("inf") or \
            thresholds[max(thresholds.keys())].up != float("inf"):
            raise ValueError("First and last thresholds must be infinity")
        #check order
        elem_list = list(thresholds.values())
        for i in range(len(elem_list)-1):
            if elem_list[i].up != elem_list[i+1].low:
                raise ValueError("Thresholds must be strictly ordered: " + str(elem_list[i].up) + ' ' + str(elem_list[i+1].low))
        for elem in elem_list:
            if elem.low >= elem.up:
                raise ValueError("Inconsistent bounds: low= " + str(elem.low) + ' up= ' + str(elem.up))
    
    @staticmethod
    def _check_matrix(states, inputs, outputs, matrix):
        """
        >>> Neuron._check_matrix(['s0', 's1'], [-1, 0], [0, 1], {'s0':{-1:0, 0:1}, 's2':{-1:0, 0:0}})
        Traceback (most recent call last):
        ...
        ValueError: Incorrect states= ['s0', 's1'], matrix states= ['s0', 's2']
        """
        #check states
        _raise_condition(set(states) == matrix.keys(), 
        "Incorrect states= " + str(states) + ", matrix states= " + str(list(matrix.keys())))
        #check inputs
        for state in states:
            _raise_condition(matrix[state].keys() == set(inputs), 
            "Incorrect inputs for state=" + str(state) + " : " + str(matrix[state]))
            _raise_condition( set(matrix[state].values()) <= set(outputs), 
            "Incorrect outputs: " + str(outputs) + ", matrix row: " + str(matrix[state]))
            #for out in matrix[state].values():
            #    _raise_condition(out in set(outputs),
            #    "Matrix Error: Output " + str(out) + " is not present. Outputs: " + str(outputs) + ", matrix row: " + str(matrix[state]))
        

    def _check_automaton(self):
        """ Checks that transition and output matrices are consistent with inputs, activity levels and states
        """
        inputs = self.thresholds.keys()
        Neuron._check_matrix(self.states, inputs, self.states, self.state_trans_matrix)
        Neuron._check_matrix(self.states, inputs, self.activity_levels, self.output_matrix)
        #check states
        #_raise_condition(set(states) == transition_matrix.keys(), 
        #"Transition matrix have incorrect states: " + "states=" + str(states) + ", matrix states=" + str(transition_matrix.keys()))
        #_raise_condition(set(states) == output_matrix.keys(), 
        #"Output matrix have incorrect states: " + "states=" + str(states) + ", matrix states=" + str(output_matrix.keys()))

def create_tonic(name, receptor_weights, transmitter_emission, activity_levels, thresholds):
    states = ["s0"]
    state_trans_matrix = {'s0':{-1:'s0', 0:'s0'}} 
    output_matrix = {'s0': {-1:0, 0:1} }
    return Neuron(name, receptor_weights, transmitter_emission, activity_levels, thresholds,
        states, state_trans_matrix, output_matrix)

def create_passive(name, receptor_weights, transmitter_emission, activity_levels, thresholds):
    states = ["s0"]
    state_trans_matrix = {'s0':{0:'s0', 1:'s0'}} 
    output_matrix = {'s0': {0:0, 1:1} }
    return Neuron(name, receptor_weights, transmitter_emission, activity_levels, thresholds,
        states, state_trans_matrix, output_matrix)

def create_delayed_passive(name, receptor_weights, transmitter_emission, activity_levels, thresholds):
    states = ["passive", "active"]
    state_trans_matrix = {'passive':{0:'passive', 1:'active'}, 
                        'active':{0:'passive', 1:'active'}}
    output_matrix = {'passive':{0:0, 1:0},
                        'active':{0:1, 1:1}}
    return Neuron(name, receptor_weights, transmitter_emission, activity_levels, thresholds,
        states, state_trans_matrix, output_matrix)

def create_oscillator(name, receptor_weights, transmitter_emission, activity_levels, thresholds, burst_duration, recharge_duration):
    assert(len(activity_levels) == 2)
    assert(len(thresholds) == 3)
    #charge states
    states = ["charge-" + str(i) for i in range(0, recharge_duration)]
    #burst states
    states.extend(["burst-" + str(i) for i in range(0, burst_duration)])
    #state and output matrices
    state_trans_matrix = {}
    output_matrix = {}
    for i in range(recharge_duration):
        state_trans_matrix[states[i]] = {-1:states[i], 0:states[i+1], 1:states[max(i+2, recharge_duration )]}
        output_matrix[states[i]] = {-1:0, 0:0, 1:0}
    for i in range(recharge_duration, recharge_duration+burst_duration):
        next_state = (i+1) % (recharge_duration+burst_duration)
        state_trans_matrix[states[i]] = {-1:states[0], 0:states[next_state], 1:states[next_state]}
        output_matrix[states[i]] = {-1:0, 0:1, 1:1}
    return Neuron(name, receptor_weights, transmitter_emission, activity_levels, thresholds,
        states, state_trans_matrix, output_matrix)

def _list_transmitters(neurons):
    #all_tr = set([n.transmitter_names() for n in neurons])
    return set([tr for n in neurons for tr in n.transmitter_names()])

class Experiment:
    def __init__(self, name, duration, neurons, transmitters, injection = None):
        """Creates an experiment
            name - the description
            duration - positive integer number. The simulation stops when number of steps reaches the duration
            neurons - list of neurons. Each neuron must have a unique name. 
            transmitters - list of transmitter names: {'ACH', 'GLU'}. 
            injection - dictionary where keys are the transmitter names and the values are lists with lengths equal the duration:
                {'ACH':[0,0,2], 'GLU':[0,1,0]} - duration=3 here
        """
        self.name = name
        if duration>0 and duration==int(duration): 
            self.duration = duration
        else:
            raise ValueError("Duration must be positive integer, duration = " + str(duration))
        self.neurons = copy.copy(neurons)
        self.transmitters = transmitters
        if injection is None:
            #injection = dict([(k,[0]*duration) for k in transmitters])
            injection = [ _zeros_dict(transmitters) ]*duration
        self.injection = injection
    
    def num_transmitters(self):
        return len(self.transmitters)

def conc_add(conc_left, conc_right):
    """ Adds to concentrations
    """
    return dict([(tr, conc_left[tr]+conc_right[tr]) for tr in conc_left])

class ModelState:
    def __init__(self, activities, concentrations, flag = True, states=[]):
        """ ModelState represents one step of a rhythm. It consists of two dictionaries:
        activities - neurons activity {'N1':1, 'N2':0}
        concentrations - transmitter concentrations {'ACH':0.5, 'GLU':0}
        flag - True if the given state is a valid combination of neurons' activity and the transmitters' concentration
        states - current internal states of the neurons. {'N1':'default', 'N2':'inhibited'}
        """
        self.activities = activities.copy()
        self.concentrations = concentrations.copy()
        self.is_valid = flag
        self.states = states

def generate_rhythm(experiment):
    """ Generate a rhythm for the specified experiment.
    Returns an object with two fields:
        activities - dict of neurons' activities [{"N1":1, "N2":0}, ...]
            len(activity) = experiment.duration
        concentrations - dict of concentrations
    """
    init_concentrations = _zeros_dict(experiment.transmitters)
    init_activities = get_activities(experiment.neurons, init_concentrations)
    history = [ ModelState(init_activities, init_concentrations) ]
    for t in range(1, experiment.duration+1):
        _update_states(experiment.neurons, history[t-1].concentrations)
        next_model_state = _concurrent_activation(experiment.neurons, history[t-1].activities, history[t-1].concentrations, experiment.injection[t-1])
        #all_branches = _recursive_activation(experiment.neurons, \
        #[history[t-1].activities], [], experiment.injection[t-1])
        #next_model_state = all_branches[0].state
        history.append( next_model_state )
        if not next_model_state.is_valid: #the concurrent activation stopped in a cycle
            return history
    return history

def get_activities(neurons, concentrations):
    """ Returns a dictionary  {neuron_name:activity} 
    
    Example: {'N1':0, 'N2': 1, 'N3': 0.5}
    """
    activities = _zeros_dict(_list_names(neurons))
    for n in neurons:
        activities[n.name] = n.get_activity(concentrations)
    return activities
  
def _concurrent_activation(neurons, initial_activity, initial_concentrations, injection):
    """ Returns ModelState object, computed according to the concurrent activation algorithm
    If the algorithm entered a loop then the returned model state is_valid flag is set to False
    """
    activities = initial_activity.copy()
    act_hist = [initial_activity]
    concentrations = initial_concentrations
    while not _is_consistent(neurons, activities, injection):
        n = _choose_neuron(neurons, activities, initial_concentrations, injection)
        concentrations = conc_add(get_concentration(neurons, activities), injection)
        activities[n.name] = n.get_activity(concentrations)
        if (len(act_hist) >= len(neurons)) and (activities in act_hist):
            return ModelState(activities, get_concentration(neurons, activities), False)
        else:
            act_hist.append(activities.copy())

    return ModelState(activities, conc_add(get_concentration(neurons, activities), injection))

def _recursive_activation(neurons, activity_history, order=[], injection=[]):
    ''' 
    TODO:
    Test, document
    '''
    Branch = namedtuple('Branch', ['order', 'state', 'activity_history'])
    initial_activity = activity_history[-1]
    concentrations = get_concentration(neurons, initial_activity, injection)
    def _update_activity(nrn):
        new_activity = initial_activity.copy()
        new_activity[nrn.name] = nrn.get_activity(concentrations)
        return new_activity
    unstable_neurons = _choose_unstable_neurons(neurons, initial_activity, injection)
    #if initial_acivity is already a consistent state then just return it
    if _is_consistent(neurons, initial_activity, injection):
        valid_final_state = ModelState(initial_activity, concentrations, True)
        return [ Branch(order, valid_final_state, activity_history ) ]
    all_branches_history = []
    for nrn in unstable_neurons:
        new_order = order + [nrn]
        new_activity = _update_activity(nrn)
        new_activity_history = activity_history + [new_activity]
        new_concentration = get_concentration(neurons, new_activity, injection)
        if (len( activity_history) >= len(neurons)) and (new_activity in  activity_history):#the algorithm enters a loop
            invalid_final_state = ModelState(new_activity, new_concentration, False)
            all_branches_history.append( Branch(new_order, invalid_final_state, new_activity_history ))
        elif _is_consistent(neurons, new_activity, injection):
            valid_final_state = ModelState(new_activity, new_concentration, True)
            all_branches_history.append( Branch(new_order, valid_final_state, new_activity_history ))
        else:
            all_branches_history = all_branches_history + \
            _recursive_activation(neurons, new_activity_history, new_order, injection)
    return all_branches_history

def _choose_unstable_neurons(neurons, activities, injection):
    concentrations = get_concentration(neurons, activities, injection)
    active_neurons = [n for n in neurons if n.get_activity(concentrations)!=activities[n.name] ]
    return active_neurons

def _choose_neuron(neurons, activities, past_concentrations, injection):
    """ Choose a neuron that should change its activity"""
    concentrations = get_concentration(neurons, activities, injection)
    #past_concentrations = get_concentration(neurons, past_activities)
    active_neurons = [n for n in neurons if n.get_activity(concentrations)!=activities[n.name] ]
    return min(active_neurons, key = lambda n: n.get_time(concentrations, past_concentrations) )
    
def _is_consistent(neurons, activities, injection):
    """ Checks whether the activities vector are valid for given neurons    
    """
    concentrations = get_concentration(neurons, activities, injection)
    for n in neurons:
        if n.get_activity(concentrations) != activities[n.name]:
            return False
    return True

def get_concentration(neurons, activities, injection = []):
    """ Returns concentration that the neurons produce under given activities
        activities - dict {'N1':1, 'N2':1.5, 'N3':0}
    """
    concentration = _zeros_dict(neurons[0].transmitter_names())
    for n in neurons:
        for tr in n.transmitter_names():
            concentration[tr] += n.transmitter_emission[tr]*activities[n.name]
    if len(injection)>0:
        concentration = conc_add(concentration, injection)
    return concentration

def _list_states(neurons):
    return dict( [ (n.name, n.current_state) for n in neurons] )
    
def _list_names(neurons):
    return [n.name for n in neurons]

def _update_states(neurons, concentrations):
    for n in neurons:
        n.next_state(concentrations)

def _zeros_dict(keys):
    return dict([(x, 0) for x in keys])

def _init_dict(keys, val):
    return dict([(x, val) for x in keys])

#def _init_list(length, val):
#    return [val]*length
   
def print_rhythm_ascii(history):
    """ 
    history - list of ModelState objects
    """
    def format_num(num):
        width = 4
        precision = 2
        return f"{decimal.Decimal(num):{width}.{precision}}"
    neuron_names = sorted([n_name for n_name in history[0].activities])
    trans_names =  sorted([tr_name for tr_name in history[0].concentrations])
    for n_name in neuron_names:#print neurons
        n_str = n_name + '\t|' + '|'.join([format_num(ms.activities[n_name]) for ms in history]).replace('|0', '| ')
        print(n_str)
    for tr_name in trans_names:
        tr_str = tr_name + '\t|' + '|'.join([format_num(ms.concentrations[tr_name]) for ms in history]).replace('|0', '| ')
        print(tr_str)


def _raise_condition(condition, message):
    if not condition:
        raise ValueError(message)

def make_configuration_space(neurons):
    '''Generates a configuration space as a graph of states and transitions.
    Returns: a named tuple (states, transitions) where states is a set of the ensemble's states. 
    An ensemble state is a named tuple (neurons_states, activities)
    A transition is a named tuple (in_state, out_state, branches)
    '''


#Test code goes below:
if __name__ == "__main__":
    INFINITY = float('inf')
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
        -2: (-INFINITY, -1),
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
    print("Tonic with PIR:")
    inhibited_conc = {'ACH':1, 'GLU':0.5}
    print(n_tonic_pir.get_activity(inhibited_conc))
    print(n_tonic_pir.next_state(inhibited_conc))
    zero_conc = {'ACH':0, 'GLU':0}
    print(n_tonic_pir.get_activity(zero_conc))
    print(n_tonic_pir.next_state(zero_conc))
    
    