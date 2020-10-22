import random

class ReplayMemory:
    """ A replay memory buffer. """

    def __init__(self, capacity):
        """ Store the 'capacity' most recent memories. """
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        """ Push a new transition. """
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        """ Uniformly sample 'batch_size' transitions. """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """ Override the len function. """
        return len(self.memory)