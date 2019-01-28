# Test for neuron.py
from neuron import Neuron
from neuron import Experiment
from neuron import generate_rhythm
from neuron import print_rhythm_ascii
from neuron import create_tonic
from neuron import create_passive
import neuron
import copy
import demo
import unittest

INFINITY = float('inf')
MINUS_INFINITY = -float('inf')


class TestRecursiveActivation(unittest.TestCase):
    def test_valid_initial_state(self):
        neurons = create_tonic_delayed_passive()
        start_activity = neuron.get_activities(
            neurons, neuron._zeros_dict(neuron._list_transmitters(neurons)))
        all_branches = neuron._recursive_activation(neurons, [start_activity])
        self.assertEqual(len(all_branches), 1)
        final_state = all_branches[0].state  # TODO: write tests
        self.assertTrue(final_state.is_valid)
        self.assertEqual(len(all_branches[0].order), 0)
        self.assertEqual(len(all_branches[0].activity_history), 1)

    def test_valid_two_branches(self):
        neurons = demo.create_hco()
        start_activity = neuron.get_activities(
            neurons, neuron._zeros_dict(neuron._list_transmitters(neurons)))
        all_branches = neuron._recursive_activation(neurons, [start_activity])
        self.assertEqual(len(all_branches), 2)
        for branch in all_branches:
            self.assertTrue(branch.state.is_valid)

    def test_invalid_state(self):
        neurons = create_incorrect_tonic_passive()
        start_activity = neuron.get_activities(
            neurons, neuron._zeros_dict(neuron._list_transmitters(neurons)))
        all_branches = neuron._recursive_activation(neurons, [start_activity])
        self.assertEqual(len(all_branches), 1)
        self.assertFalse(all_branches[0].state.is_valid)


class TestNeuron(unittest.TestCase):
    def test_neuron(self):
        n_list = demo.create_hco()
        inhibited_conc = {'ACH': 1, 'GLU': 0.5}
        self.assertEqual(n_list[0].get_activity(inhibited_conc), 0)
        self.assertEqual(n_list[0].next_state(inhibited_conc), 'inh')
        zero_conc = {'ACH': 0, 'GLU': 0}
        self.assertEqual(n_list[0].get_current_state(), 'inh')
        self.assertEqual(n_list[0].get_activity(zero_conc), 2)
        self.assertEqual(n_list[0].next_state(zero_conc), 'act')


def create_incorrect_tonic_passive():
    # receptors
    receptors_tonic = {'ACH': -1, 'GLU': 0}
    receptors_passive = {'ACH': 0, 'GLU': 1}
    # emission
    emission_tonic = {'ACH': 0, 'GLU': 1}
    emission_passive = {'ACH': 1, 'GLU': 0}
    # thresholds
    thresholds_tonic = {-1: (MINUS_INFINITY, -0.5), 0: (-0.5, INFINITY)}
    thresholds_passive = {0: (MINUS_INFINITY, 0.5), 1: (0.5, INFINITY)}
    activity_levels = (0, 1)
    n_tonic = create_tonic('N1', receptors_tonic,
                           emission_tonic, activity_levels,  thresholds_tonic)
    n_passive = create_passive(
        'N2', receptors_passive, emission_passive, activity_levels,  thresholds_passive)
    return [n_tonic, n_passive]


def create_tonic_delayed_passive():
    # receptors
    receptors_tonic = {'ACH': -1, 'GLU': 0}
    receptors_passive = {'ACH': 0, 'GLU': 1}
    # emission
    emission_tonic = {'ACH': 0, 'GLU': 1}
    emission_passive = {'ACH': 1, 'GLU': 0}
    # thresholds
    thresholds_tonic = {-1: (MINUS_INFINITY, -0.5), 0: (-0.5, INFINITY)}
    thresholds_passive = {0: (MINUS_INFINITY, 0.5), 1: (0.5, INFINITY)}
    activity_levels = (0, 1)
    n_tonic = create_tonic('N1', receptors_tonic,
                           emission_tonic, activity_levels,  thresholds_tonic)
    n_passive = neuron.create_delayed_passive(
        'N2', receptors_passive, emission_passive, activity_levels,  thresholds_passive)
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


def test_hco():
    neurons = demo.create_hco()
    neurons[1].activity_levels = (0, 0.92, 2)
    neurons[0].activity_levels = (0, 0.91, 2)
    exp = Experiment("Test experiment", 10, neurons, ['ACH', 'GLU'])
    hist = generate_rhythm(exp)
    assert(hist[-1].is_valid)
    max_name = (min(neurons, key=lambda t: -t.activity_levels[1])).name
    min_name = (min(neurons, key=lambda t: t.activity_levels[1])).name
    assert(hist[1].activities[max_name] >
           0 and hist[1].activities[min_name] == 0)
    print("Half-center oscillator:")
    print_rhythm_ascii(hist)


def test_oscillator():
    receptors = {'ACH': -1, 'GLU': 0}
    # emission
    emission = {'ACH': 0, 'GLU': 1}
    # thresholds
    thresholds = {-1: (MINUS_INFINITY, -0.5),
                  0: (-0.5, 0.5), 1: (0.5, INFINITY)}
    activity_levels = (0, 1)
    burst_duration = 2
    recharge_duration = 3
    n = neuron.create_oscillator(
        "N1", receptors, emission, activity_levels, thresholds, burst_duration, recharge_duration)
    assert(len(n.states) == burst_duration+recharge_duration)
    exp = Experiment("Test experiment", 10, [n], ['ACH', 'GLU'])
    hist = generate_rhythm(exp)
    print("Oscillator:")
    print_rhythm_ascii(hist)


if __name__ == "__main__":
    # test_neuron()
    # test_hco()
    # test_loop_concurrent_activation()
    # test_tonic_delayed_passive()
    # test_oscillator()
    unittest.main(verbosity=2, exit=False)
