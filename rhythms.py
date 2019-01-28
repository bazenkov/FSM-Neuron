from collections import namedtuple

Burst = namedtuple('Burst', ['neuron_name', 'onset', 'duration'])

class Rhythm:
    """List of bursts with methods for rhythm features.
	Properties:
	neurons - list of neurons' names. Like ['N1', 'N2', 'N3'];
	bursts - list of bursts. A burst is a triple: (neuron_name, onset, duration);
	period - duration of the rhythm's one cycle.
	"""
    def __init__(self, neurons = [], period = 0, bursts = []):
        self.neurons = neurons
        self.period = period
        self.bursts = bursts

    def num_of_neurons(self):
        return len(self.neurons)

    def get_bursts(self, neuron_name = ''):
        """Get list of all bursts of the specified neuron."""
        if len(neuron_name) == 0:
            return self.bursts
        else:
            return [b for b in self.bursts if get_neuron_name(b) == neuron_name]

    def get_burst(self, neuron_name, burst_num=0):
        burst_list = self.get_bursts(neuron_name)
        return burst_list[burst_num]

    def phase_on(self, burst):
        return onset(burst)/self.period
    
    def phase_off(self, burst):
        return offset(burst)/self.period

def onset(burst):
    return burst[1]
    
def get_neuron_name(burst):
    return burst[0]
    
def duration(burst):
    return burst[2]

def offset(burst):
    return onset(burst) + duration(burst)
    
def extract_rhythm(history):
    """ Extracts rhythm from the given history of neural activity.
    history - a list of neuron.ModelState objects
    """


#Test code goes below:
if __name__ == "__main__":
    r = Rhythm()
    r.neurons = ['N1', 'N2']
    r.period = 3
    r.bursts = [['N1', 0, 1], ['N2', 1.5, 1], ['N3', 1, 2], ['N1', 1.5, 1]]

    print(r.neurons)
    print(r.period)
    print(r.get_bursts())
    print('N1 bursts are ' + str(r.get_bursts('N1')))
    print('N2 bursts are ' + str(r.get_bursts('N2')))
    print(r.get_bursts('N1')[0])

    print('N1 first onset is ' + str(onset(r.get_bursts('N1')[0])))
    print('N1 first offset is ' + str(offset(r.get_bursts('N1')[0])))
    print('N1 second onset is ' + str(onset(r.get_bursts('N1')[1])))
    print('N2 onset is ' + str(onset(r.get_bursts('N2')[0])))
    print('N2 offset is ' + str(offset(r.get_bursts('N2')[0])))
    print('N1 phase onset is ' + str(r.phase_on(r.get_bursts('N1')[0])))
    print('N2 phase onset is ' + str(r.phase_on(r.get_bursts('N2')[0])))
    print('N1 second burst is ' + str(r.get_burst('N1',1)))