from __future__ import print_function
import neat

# Dummy NEAT data
config_path = 'C:\\Users\\vyshn\\OneDrive\\Desktop\\Documents\\flappy_bird\\configuration-ff.txt'
genome = neat.DefaultGenome(1)
pop = neat.Population(config_path)

# Create dummy statistics
class DummyStatistics:
    def __init__(self, most_fit_genomes):
        self.most_fit_genomes = most_fit_genomes

    def get_fitness_mean(self):
        return [genome.fitness for genome in self.most_fit_genomes]

    def get_fitness_stdev(self):
        return [0.1] * len(self.most_fit_genomes)  # Dummy standard deviation

# Create dummy statistics with 10 generations
dummy_genomes = [genome] * 10
dummy_stats = DummyStatistics(dummy_genomes)

# Import the provided functions
from neat_visualization import plot_stats, plot_species, draw_net

# Example usage of functions
plot_stats(dummy_stats, view=True, filename='avg_fitness.png')
plot_species(dummy_stats, view=True, filename='speciation.png')

# Draw the network of the first genome
draw_net(pop.config, dummy_genomes[0], view=True, filename='neural_network')
