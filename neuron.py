""" Neuron as a finite state machine """
class Neuron:

    def __init__(self, receptor_weights,
        transmitter_emission, activity_levels, thresholds,
        states, state_trans_matrix, output_matrix):
        """ Properties:
            - receptor_weights {'ACH':(-1, -0.5), 'GLU':(0, 0)}
			- transmitter_emission {'ACH':0, 'GLU':1}
			- activity_levels (0, 1, 2)
			- thresholds ((-1, -0.5), (0, 0.5), (1, INFINITY)
			- states ('s0', 's1')
			- state_trans_matrix {'s0':{-1:'s1', 0:'s0', 1:'s0'}, 's1':{-1:'s1', 0:'s0', 1:'s0'}} 
			- output_matrix {'s0':{-1:0, 0:1, 1:2}, 's1':{-1:0, 0:0, 1:0}}
            - current state
        """
        self.receptor_weights = receptor_weights	
        self.max_delay = 0
        self.transmitter_emission = transmitter_emission
        self.activity_levels = activity_levels
        self.thresholds	= thresholds
        self.states = states
        self.state_trans_matrix = state_trans_matrix
        self.output_matrix = output_matrix	
        self.current_state = states[0]
        #self.current_activity = self.get_activity([0]*len(receptor_weights))
#read from a file
#get next state for the concentrations
    def get_total_input(self, concentrations):
        """ concentrations - dict {'ACH':0, 'GLU':0.5} """
        delay = 0
        return sum(concentrations[trans]*self.receptor_weights[trans][delay] 
            for trans in self.receptor_weights)

    def next_state(self, concentrations):	 
        discretized = self.discrete_input(self.get_total_input(concentrations))
        self.current_state = self.state_trans_matrix[self.current_state][discretized]
        return self.get_current_state()
    
    def get_activity(self, concentrations):
        """ get next activity for the concentrations """
        total_input = self.get_total_input(concentrations)
        discretized = self.discrete_input(total_input)
        return self.output_matrix[self.get_current_state()][discretized]

    def discrete_input(self, total_input):
        """Get discretized input"""    
        for tr in self.thresholds:
            if total_input < tr[1]:
                return tr[0]
        print("ERROR: incorrect discretization")

    def get_current_state(self):
        return self.current_state


#Test code goes below:
if __name__ == "__main__":
    INFINITY = float('inf')
    n_tonic_pir = Neuron({'ACH':(-1, -0.5), 'GLU':(0, 0)}, #receptor_weights 
    {'ACH':0, 'GLU':1}, #emission
    (0, 1, 2), #activity levels
    ((-1, -0.5), (0, 0.5), (1, INFINITY)), #thresholds
    ('s0', 's1'), #states
    {'s0':{-1:'s1', 0:'s0', 1:'s0'}, 's1':{-1:'s1', 0:'s0', 1:'s0'}}, #state_trans_matrix 
    {'s0':{-1:0, 0:1, 1:2}, 's1':{-1:0, 0:2, 1:2}})
    print("Tonic with PIR:")
    inhibited_conc = {'ACH':1, 'GLU':0.5}
    print(n_tonic_pir.get_activity(inhibited_conc))
    print(n_tonic_pir.next_state(inhibited_conc))
    zero_conc = {'ACH':0, 'GLU':0}
    print(n_tonic_pir.get_activity(zero_conc))
    print(n_tonic_pir.next_state(zero_conc))