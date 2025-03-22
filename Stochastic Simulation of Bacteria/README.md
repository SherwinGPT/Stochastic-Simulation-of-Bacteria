      
# Stochastic Simulation of Bacteria

TLDR: 
1. Install the requirements in the Requirements.txt file.
2. Run `bacteria_simulation.py` for simulations with and without antibiotics.
3. Run `test_bacteria_simulation.py` to execute unit tests.


## Description
This project simulates the growth of bacteria populations in a patient, both with and without the introduction of antibiotics. 
It models simple bacteria, resistant bacteria, and the effects of mutation and population density on reproduction and survival. 
It includes visualizations of population dynamics over time.


## Table of Contents
- [Stochastic Simulation of Bacteria](#stochastic-simulation-of-bacteria)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Running the Simulation without Antibiotics](#running-the-simulation-without-antibiotics)
    - [Running the Simulation with Antibiotics](#running-the-simulation-with-antibiotics)
    - [Running the Unit Tests](#running-the-unit-tests)
  - [Classes](#classes)
    - [Exception Classes](#exception-classes)
    - [Bacteria Classes](#bacteria-classes)
    - [Patient Classes](#patient-classes)
  - [Functions](#functions)


## Project Structure
```
Bacteria Simulation/
├── bacteria_simulation.py       # Main simulation logic and classes
├── test_bacteria_simulation.py  # Unit tests for the simulation
├── README.md                    # This file
├── requirements.txt             # Project dependencies
└── .gitignore                   # Files to ignore
```


## Installation
1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SherwinGPT/Stochastic-Simulation-of-Bacteria.git
    cd Stochastic-Simulation-of-Bacteria
    ```

2.  **Create and activate a virtual environment (highly recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # OR
    .venv\Scripts\activate  # On Windows (cmd.exe)
    # .venv\Scripts\Activate.ps1  # On Windows (powershell)
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Running the Simulation without Antibiotics
The `simulation_without_antibiotic` function runs the simulation without introducing any antibiotics.  It takes several 
parameters to control the simulation:

*   `num_bacteria`: The initial number of bacteria.
*   `max_pop`: The maximum carrying capacity of the environment.
*   `birth_prob`: The maximum probability of a bacterium reproducing.
*   `death_prob`: The maximum probability of a bacterium dying.
*   `num_trials`: The number of independent simulation trials to run.

Example usage (this is already included in `bacteria_simulation.py`):

```python
import bacteria_simulation

populations = bacteria_simulation.simulation_without_antibiotic(
    num_bacteria=100,
    max_pop=1000,
    birth_prob=0.1,
    death_prob=0.025,
    num_trials=50
)
```

To run this simulation, execute:

```bash
python bacteria_simulation.py
```


### Running the Simulation with Antibiotics
The `simulation_with_antibiotic` function simulates the introduction of an antibiotic after a certain number of time steps. 
It takes the following additional parameters:

*   `resistant`: Whether the initial bacteria are resistant to the antibiotic.
*   `mut_prob`: The probability of a non-resistant bacterium mutating to become resistant.

Example usage (also included in `bacteria_simulation.py`):

```python      
import bacteria_simulation

total_pop, resistant_pop = bacteria_simulation.simulation_with_antibiotic(
    num_bacteria=100,
    max_pop=1000,
    birth_prob=0.3,
    death_prob=0.2,
    resistant=False,
    mut_prob=0.8,
    num_trials=50
)
```
    
To run this simulation, execute:

```bash      
python bacteria_simulation.py
```


### Running the Unit Tests
To run the unit tests, execute:

```bash     
python test_bacteria_simulation.py
```


## Classes
### Exception Classes
*   `NoChildException`: Raised when reproduce() does not create an offspring.


### Bacteria Classes
*   `SimpleBacteria(birth_prob, death_prob)`: Represents a bacterium without antibiotic resistance.

*   `ResistantBacteria(birth_prob, death_prob, resistant, mut_prob)`: Represents a bacterium that can have antibiotic 
    resistance and can mutate.


### Patient Classes
*   `Patient(bacteria, max_pop)`: Represents a patient not taking antibiotics.

*   `TreatedPatient(bacteria, max_pop)`: Represents a patient who can be treated with antibiotics.


## Functions
*   `make_one_curve_plot(x_coords, y_coords, x_label, y_label, title)`: Creates a plot with one curve.

*   `make_two_curve_plot(x_coords, y_coords1, y_coords2, y_name1, y_name2, x_label, y_label, title)`: Creates a plot with two curves.

*   `calc_pop_avg(populations, n)`: Calculates the average bacteria population size across trials at time step n.

*   `simulation_without_antibiotic(num_bacteria, max_pop, birth_prob, death_prob, num_trials)`: Runs the simulation without antibiotics.

*   `calc_pop_std(populations, t)`: Calculates the standard deviation of bacteria populations at time step t.

*   `calc_95_ci(populations, t)`: Calculates the 95% confidence interval for the bacteria population at time step t.

*   `simulation_with_antibiotic(num_bacteria, max_pop, birth_prob, death_prob, resistant, mut_prob, num_trials)`: Runs the simulation with 
    an antibiotic introduced.

*   `compute_confidence_interval(data)`: Computes the 95% confidence interval.