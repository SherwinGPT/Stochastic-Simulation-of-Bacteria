import math
import numpy as np
import pylab as pl
import random


class NoChildException(Exception):
    """
    NoChildException is raised by the reproduce() method in the SimpleBacteria
    and ResistantBacteria classes to indicate that a bacteria cell does not
    reproduce.
    """


def make_one_curve_plot(x_coords, y_coords, x_label, y_label, title):
    """
    Makes a plot of the x coordinates and the y coordinates with the labels
    and title provided.

    Args:
        x_coords (list of floats): x coordinates to graph
        y_coords (list of floats): y coordinates to graph
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): title for the graph
    """
    pl.figure()
    pl.plot(x_coords, y_coords)
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.title(title)
    pl.show()


def make_two_curve_plot(x_coords,
                        y_coords1,
                        y_coords2,
                        y_name1,
                        y_name2,
                        x_label,
                        y_label,
                        title):
    """
    Makes a plot with two curves on it, based on the x coordinates with each of
    the set of y coordinates provided.

    Args:
        x_coords (list of floats): the x coordinates to graph
        y_coords1 (list of floats): the first set of y coordinates to graph
        y_coords2 (list of floats): the second set of y-coordinates to graph
        y_name1 (str): name describing the first y-coordinates line
        y_name2 (str): name describing the second y-coordinates line
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): the title of the graph
    """
    pl.figure()
    pl.plot(x_coords, y_coords1, label=y_name1)
    pl.plot(x_coords, y_coords2, label=y_name2)
    pl.legend()
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.title(title)
    pl.show()


class SimpleBacteria(object):
    """A simple bacteria cell with no antibiotic resistance"""

    def __init__(self, birth_prob, death_prob):
        """
        Args:
            birth_prob (float in [0, 1]): Maximum possible reproduction
                probability
            death_prob (float in [0, 1]): Maximum death probability
        """
        self.birth_prob = birth_prob
        self.death_prob = death_prob

    def is_killed(self):
        """
        Stochastically determines whether this bacteria cell is killed in
        the patient's body at a time step, i.e. the bacteria cell dies with
        some probability equal to the death probability each time step.

        Returns:
            bool: True with probability self.death_prob, False otherwise.
        """
        return random.random() <= self.death_prob

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the Patient and
        TreatedPatient classes.

        The bacteria cell reproduces with probability
        self.birth_prob * (1 - pop_density).

        If this bacteria cell reproduces, then reproduce() creates and returns
        the instance of the offspring SimpleBacteria (which has the same
        birth_prob and death_prob values as its parent).

        Args:
            pop_density (float): The population density, defined as the
                current bacteria population divided by the maximum population

        Returns:
            SimpleBacteria: A new instance representing the offspring of
                this bacteria cell (if the bacteria reproduces). The child
                should have the same birth_prob and death_prob values as
                this bacteria.

        Raises:
            NoChildException if this bacteria cell does not reproduce.
        """
        if random.random() <= self.birth_prob * (1 - pop_density):
            return SimpleBacteria(self.birth_prob, self.death_prob)
        else:
            raise NoChildException()


class Patient(object):
    """
    Representation of a simplified patient. The patient does not take any
    antibiotics and his/her bacteria populations have no antibiotic resistance.
    """
    def __init__(self, bacteria, max_pop):
        """
        Args:
            bacteria (list of SimpleBacteria): The bacteria in the population
            max_pop (int): Maximum possible bacteria population size for
                this patient
        """
        self.bacteria = bacteria
        self.max_pop = max_pop

    def get_total_pop(self):
        """
        Gets the size of the current total bacteria population.

        Returns:
            int: The total bacteria population
        """
        return len(self.bacteria)

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute the following steps in
        this order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. Calculate the current population density by dividing the surviving
           bacteria population by the maximum population. This population
           density value is used for the following steps until the next call
           to update()

        3. Based on the population density, determines whether each surviving
           bacteria cell should reproduce and add offspring bacteria cells to
           a list of bacteria in this patient. New offspring do not reproduce.

        4. Reassigns the patient's bacteria list to be the list of surviving
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        surviving_bacteria = [b for b in self.bacteria if not b.is_killed()]
        pop_density = len(surviving_bacteria) / self.max_pop
        offspring = []

        for b in surviving_bacteria:
            try:
                offspring.append(b.reproduce(pop_density))
            except NoChildException:
                continue

        self.bacteria = surviving_bacteria + offspring
        return len(self.bacteria)


def calc_pop_avg(populations, n):
    """
    Finds the average bacteria population size across trials at time step n

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j

    Returns:
        float: The average bacteria population size at time step n
    """
    total = sum(trial[n] for trial in populations)
    return total / len(populations)


def simulation_without_antibiotic(num_bacteria,
                                  max_pop,
                                  birth_prob,
                                  death_prob,
                                  num_trials):
    """
    Runs the simulation and plots the graph. No antibiotics
    are used, and bacteria do not have any antibiotic resistance.

    For each of num_trials trials:
        * instantiate a list of SimpleBacteria
        * instantiate a Patient using the list of SimpleBacteria
        * simulate changes to the bacteria population for 300 timesteps,
          recording the bacteria population after each time step. The first 
          time step should contain the starting number of bacteria in the patient

    Then, plots the average bacteria population size (y-axis) as a function of
    elapsed time steps (x-axis)

    Args:
        num_bacteria (int): number of SimpleBacteria to create for patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float in [0, 1]): maximum reproduction
            probability
        death_prob (float in [0, 1]): maximum death probability
        num_trials (int): number of simulation runs to execute

    Returns:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j
    """
    populations = []
    num_timesteps = 300

    for _ in range(num_trials):
        # Initialize bacteria and patient
        bacteria = [SimpleBacteria(birth_prob, death_prob) for _ in range(num_bacteria)]
        patient = Patient(bacteria, max_pop)

        # Simulate 300 time steps
        trial_population = []
        for _ in range(num_timesteps):
            trial_population.append(patient.update())

        populations.append(trial_population)

    # Calculate average population over time
    avg_populations = [calc_pop_avg(populations, t) for t in range(num_timesteps)]

    # Plot the results
    make_one_curve_plot(
        x_coords=list(range(num_timesteps)),
        y_coords=avg_populations,
        x_label="Time Steps",
        y_label="Average Population",
        title="Bacteria Population Over Time (No Antibiotics)"
    )

    return populations


# To run the simulation, uncomment the line below
populations = simulation_without_antibiotic(100, 1000, 0.1, 0.025, 50)


def calc_pop_std(populations, t):
    """
    Finds the standard deviation of populations across different trials
    at time step t by:
        * calculating the average population at time step t
        * compute average squared distance of the data points from the average
          and take its square root
    
    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        float: the standard deviation of populations across different trials at
             a specific time step
    """
    # Step 1: Calculate the mean population at time step t
    mean_population = calc_pop_avg(populations, t)

    # Step 2: Calculate the squared differences from the mean for each trial
    squared_diffs = [(trial[t] - mean_population) ** 2 for trial in populations]

    # Step 3: Calculate the variance
    variance = sum(squared_diffs) / len(populations)

    # Step 4: Return the square root of the variance (standard deviation)
    return variance ** 0.5


def calc_95_ci(populations, t):
    """
    Finds a 95% confidence interval around the average bacteria population
    at time t by:
        * computing the mean and standard deviation of the sample
        * using the standard deviation of the sample to estimate the
          standard error of the mean (SEM)
        * using the SEM to construct confidence intervals around the
          sample mean

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        mean (float): the sample mean
        width (float): 1.96 * SEM

        Returns a tuple containing (mean, width)
    """
    # Step 1: Calculate the sample mean
    mean_population = calc_pop_avg(populations, t)

    # Step 2: Calculate the standard deviation
    std_dev = calc_pop_std(populations, t)

    # Step 3: Calculate the standard error of the mean (SEM)
    num_trials = len(populations)
    sem = std_dev / (num_trials ** 0.5)

    # Step 4: Calculate the 95% confidence interval width (1.96 * SEM)
    ci_width = 1.96 * sem

    # Step 5: Return the mean and the CI width
    return (mean_population, ci_width)


class ResistantBacteria(SimpleBacteria):
    """A bacteria cell that can have antibiotic resistance."""

    def __init__(self, birth_prob, death_prob, resistant, mut_prob):
        """
        Args:
            birth_prob (float in [0, 1]): reproduction probability
            death_prob (float in [0, 1]): death probability
            resistant (bool): whether this bacteria has antibiotic resistance
            mut_prob (float): mutation probability for this
                bacteria cell. This is the maximum probability of the
                offspring acquiring antibiotic resistance
        """
        # Initialize the SimpleBacteria part (inherits from SimpleBacteria)
        super().__init__(birth_prob, death_prob)
        self.resistant = resistant
        self.mut_prob = mut_prob

    def get_resistant(self):
        """Returns whether the bacteria has antibiotic resistance"""
        return self.resistant

    def is_killed(self):
        """Stochastically determines whether this bacteria cell is killed in
        the patient's body at a given time step.

        Checks whether the bacteria has antibiotic resistance. If resistant,
        the bacteria dies with the regular death probability. If not resistant,
        the bacteria dies with the regular death probability / 4.

        Returns:
            bool: True if the bacteria dies with the appropriate probability
                and False otherwise.
        """
        # If the bacteria is resistant, it dies with the usual death_prob
        if self.resistant:
            return random.random() < self.death_prob
        # If it's not resistant, it dies with 1/4 the normal death_prob
        else:
            return random.random() < self.death_prob / 4

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the TreatedPatient class.

        A surviving bacteria cell will reproduce with probability:
        self.birth_prob * (1 - pop_density).

        If the bacteria cell reproduces, then reproduce() creates and returns
        an instance of the offspring ResistantBacteria, which will have the
        same birth_prob, death_prob, and mut_prob values as its parent.

        If the bacteria has antibiotic resistance, the offspring will also be
        resistant. If the bacteria does not have antibiotic resistance, its
        offspring have a probability of self.mut_prob * (1-pop_density) of
        developing that resistance trait. That is, bacteria in less densely
        populated environments have a greater chance of mutating to have
        antibiotic resistance.

        Args:
            pop_density (float): the population density

        Returns:
            ResistantBacteria: an instance representing the offspring of
            this bacteria cell (if the bacteria reproduces). The child should
            have the same birth_prob, death_prob values and mut_prob
            as this bacteria. Otherwise, raises a NoChildException if this
            bacteria cell does not reproduce.
        """
        if random.random() < self.birth_prob * (1 - pop_density):
            # If the bacteria is resistant, the offspring will also be resistant
            if self.resistant:
                return ResistantBacteria(self.birth_prob, self.death_prob, True, self.mut_prob)
            else:
                # If not resistant, offspring may mutate with a certain probability
                if random.random() < self.mut_prob * (1 - pop_density):
                    return ResistantBacteria(self.birth_prob, self.death_prob, True, self.mut_prob)
                else:
                    return ResistantBacteria(self.birth_prob, self.death_prob, False, self.mut_prob)

        else:
            raise NoChildException("No offspring this time")


class TreatedPatient(Patient):
    """
    Representation of a treated patient. The patient is able to take an
    antibiotic and his/her bacteria population can acquire antibiotic
    resistance. The patient cannot go off an antibiotic once on it.
    """
    def __init__(self, bacteria, max_pop):
        """
        Args:
            bacteria: The list representing the bacteria population (a list of
                      bacteria instances)
            max_pop: The maximum bacteria population for this patient (int)

        This function initializes self.on_antibiotic, which represents
        whether a patient has been given an antibiotic. Initially, the
        patient has not been given an antibiotic.        
        """
        super().__init__(bacteria, max_pop)
        self.on_antibiotic = False  # Initially, the patient is not on antibiotics

    def set_on_antibiotic(self):
        """
        Administer an antibiotic to this patient. The antibiotic acts on the
        bacteria population for all subsequent time steps.
        """
        self.on_antibiotic = True

    def get_resist_pop(self):
        """
        Get the population size of bacteria cells with antibiotic resistance

        Returns:
            int: the number of bacteria with antibiotic resistance
        """
        return sum(1 for bacterium in self.bacteria if bacterium.get_resistant())

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute these actions in order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. If the patient is on antibiotics, the surviving bacteria cells from
           (1) only survive further if they are resistant. If the patient is
           not on the antibiotic, keep all surviving bacteria cells from (1)

        3. Calculate the current population density. This value is used until
           the next call to update(). Same calculations used as in Patient

        4. Based on this value of population density, determines whether each
           surviving bacteria cell should reproduce and adds offspring bacteria
           cells to the list of bacteria in this patient.

        5. Reassign the patient's bacteria list to be the list of survived
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        surviving_bacteria = []

        # Step 1: Check if each bacteria cell dies
        for bacterium in self.bacteria:
            if not bacterium.is_killed():
                surviving_bacteria.append(bacterium)

        # Step 2: If the patient is on antibiotics, only resistant bacteria survive
        if self.on_antibiotic:
            surviving_bacteria = [b for b in surviving_bacteria if b.get_resistant()]

        # Step 3: Calculate population density
        pop_density = len(surviving_bacteria) / self.max_pop

        # Step 4: Reproduce bacteria based on population density
        new_bacteria = []
        for bacterium in surviving_bacteria:
            try:
                offspring = bacterium.reproduce(pop_density)
                new_bacteria.append(offspring)
            except NoChildException:
                pass

        # Step 5: Update the bacteria list with survivors and new offspring
        self.bacteria = surviving_bacteria + new_bacteria

        return len(self.bacteria)


def simulation_with_antibiotic(num_bacteria,
                               max_pop,
                               birth_prob,
                               death_prob,
                               resistant,
                               mut_prob,
                               num_trials):
    """
    Runs simulations and plots the graphs.

    For each of num_trials trials:
        * instantiate a list of ResistantBacteria
        * instantiate a patient
        * runs a simulation for 150 timesteps, adds the antibiotic, and runs 
          the simulation for an additional 250 timesteps, recording the total
          bacteria population and the resistance bacteria population after
          each time step

    Plot the average bacteria population size for both the total bacteria
    population and the antibiotic-resistant bacteria population (y-axis) as a
    function of elapsed time steps (x-axis) on the same plot.

    Args:
        num_bacteria (int): number of ResistantBacteria to create for
            the patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float int [0-1]): reproduction probability
        death_prob (float in [0, 1]): probability of a bacteria cell dying
        resistant (bool): whether the bacteria initially have
            antibiotic resistance
        mut_prob (float in [0, 1]): mutation probability for the
            ResistantBacteria cells
        num_trials (int): number of simulation runs to execute

    Returns: a tuple of two lists of lists, or two 2D arrays
        populations (list of lists or 2D array): the total number of bacteria
            at each time step for each trial; total_population[i][j] is the
            total population for trial i at time step j
        resistant_pop (list of lists or 2D array): the total number of
            resistant bacteria at each time step for each trial;
            resistant_pop[i][j] is the number of resistant bacteria for
            trial i at time step j
    """
    populations = []  # Total bacteria population at each timestep for each trial
    resistant_pop = []  # Resistant bacteria population at each timestep for each trial

    for trial in range(num_trials):
        # Initialize the bacteria population and patient
        bacteria = [ResistantBacteria(birth_prob, death_prob, resistant, mut_prob) for _ in range(num_bacteria)]
        patient = TreatedPatient(bacteria, max_pop)

        trial_total_pop = []  # Total population for this trial
        trial_resist_pop = []  # Resistant population for this trial

        # Phase 1: No antibiotic for 150 timesteps
        for _ in range(150):
            trial_total_pop.append(patient.update())
            trial_resist_pop.append(patient.get_resist_pop())

        # Administer antibiotic
        patient.set_on_antibiotic()

        # Phase 2: With antibiotic for 250 timesteps
        for _ in range(250):
            trial_total_pop.append(patient.update())
            trial_resist_pop.append(patient.get_resist_pop())

        populations.append(trial_total_pop)
        resistant_pop.append(trial_resist_pop)

    # Compute the average population sizes for plotting
    avg_total_pop = [sum([populations[trial][t] for trial in range(num_trials)]) / num_trials for t in range(400)]
    avg_resist_pop = [sum([resistant_pop[trial][t] for trial in range(num_trials)]) / num_trials for t in range(400)]

    # Generate the plot
    make_two_curve_plot(
        list(range(400)),
        avg_total_pop,
        avg_resist_pop,
        "Total Population",
        "Resistant Population",
        "Timestep",
        "Average Population",
        "Bacteria Simulation with Antibiotic"
    )

    return populations, resistant_pop


# To run the simulations, uncomment the next lines one at a time
total_pop, resistant_pop = simulation_with_antibiotic(num_bacteria=100,
                                                      max_pop=1000,
                                                      birth_prob=0.3,
                                                      death_prob=0.2,
                                                      resistant=False,
                                                      mut_prob=0.8,
                                                      num_trials=50)

#total_pop, resistant_pop = simulation_with_antibiotic(num_bacteria=100,
#                                                      max_pop=1000,
#                                                      birth_prob=0.17,
#                                                      death_prob=0.2,
#                                                      resistant=False,
#                                                      mut_prob=0.8,
#                                                      num_trials=50)


def compute_confidence_interval(data):
    """
    Computes the 95% confidence interval for the given data.
    
    Args:
        data (list): List of data points (e.g., population at timestep 299 for all trials)
        
    Returns:
        tuple: (mean, lower bound of CI, upper bound of CI)
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    margin_of_error = round(1.96 * (std_dev / np.sqrt(len(data))), 2)
    return mean, mean - margin_of_error, mean + margin_of_error

# Example usage for timestep 299:
timestep = 299
total_populations_at_timestep = [pop[timestep] for pop in total_pop]
resistant_populations_at_timestep = [pop[timestep] for pop in resistant_pop]

# Compute CIs
total_ci = compute_confidence_interval(total_populations_at_timestep)
resistant_ci = compute_confidence_interval(resistant_populations_at_timestep)

print(f"Total Population CI at timestep 299 (mean, lower, upper): {total_ci}")
print(f"Resistant Population CI at timestep 299 (mean, lower, upper): {resistant_ci}")
