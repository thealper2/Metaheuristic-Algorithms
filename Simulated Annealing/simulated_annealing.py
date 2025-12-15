import math
import random
from typing import TypeVar, Callable, Optional, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum
import time
from abc import ABC, abstractmethod

T = TypeVar("T")


class CoolingSchedule(Enum):
    """Cooling schedule types"""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    QUADRATIC = "quadratic"


@dataclass
class SimulatedAnnealingResult:
    """Simulated annealing optimization result"""

    best_solution: Any
    best_energy: float
    energies_history: List[float]
    temperatures_history: List[float]
    execution_time: float
    iterations: int
    accepted_solutions: int
    acceptance_rate: float


@dataclass
class SAConfig:
    """Configuration for Simulated Annealing"""

    initial_temp: float = 1000.0
    min_temp: float = 1e-8
    cooling_rate: float = 0.95
    max_iterations: int = 10000
    cooling_schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL
    metropolis_criterion: bool = True  # Use Metropolis criterion for acceptance
    adaptive_cooling: bool = False  # Adapt cooling based on acceptance rate
    restart_threshold: float = (
        0.01  # Restart if improvement < threshold for N iterations
    )

    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.initial_temp <= 0:
            raise ValueError("Initial temperature must be positive")
        if self.min_temp <= 0:
            raise ValueError("Minimum temperature must be positive")
        if self.cooling_rate <= 0 or self.cooling_rate >= 1:
            raise ValueError("Cooling rate must be between 0 and 1")
        if self.max_iterations <= 0:
            raise ValueError("Maximum iterations must be positive")


class SolutionGenerator(ABC):
    """Abstract base class for solution generation"""

    @abstractmethod
    def generate_initial(self) -> Any:
        """Generate initial solution"""
        pass

    @abstractmethod
    def generate_neighbor(self, current_solution: Any, temperature: float) -> Any:
        """Generate neighbor solution"""
        pass

    @abstractmethod
    def calculate_energy(self, solution: Any) -> float:
        """Calculate energy (cost) of solution"""
        pass


class SimulatedAnnealing:
    """
    Simulated Annealing Algorithm Implementation
    """

    def __init__(self, config: Optional[SAConfig] = None):
        """
        Initialize Simulated Annealing algorithm.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or SAConfig()
        self.config.validate()

        # Statistics
        self.stats = {
            "total_iterations": 0,
            "accepted_worse": 0,
            "accepted_better": 0,
            "restarts": 0,
        }

    def _acceptance_probability(
        self, current_energy: float, new_energy: float, temperature: float
    ) -> float:
        """
        Calculate acceptance probability for a new solution.

        Args:
            current_energy: Energy of current solution
            new_energy: Energy of new solution
            temperature: Current temperature

        Returns:
            Acceptance probability between 0 and 1
        """
        energy_delta = new_energy - current_energy

        # Always accept better solutions
        if energy_delta < 0:
            return 1.0

        # For worse solutions, use Metropolis criterion if enabled
        if self.config.metropolis_criterion:
            # Avoid division by zero
            if temperature < 1e-10:
                return 0.0
            return math.exp(-energy_delta / temperature)

        return 0.0

    def _cool_temperature(
        self, temperature: float, iteration: int, acceptance_rate: float = 0.5
    ) -> float:
        """
        Cool the temperature according to selected schedule.

        Args:
            temperature: Current temperature
            iteration: Current iteration number
            acceptance_rate: Current acceptance rate for adaptive cooling

        Returns:
            New temperature
        """
        if self.config.adaptive_cooling:
            # Adjust cooling rate based on acceptance rate
            adaptive_rate = self.config.cooling_rate
            if acceptance_rate < 0.2:
                # Slow down cooling if acceptance is too low
                adaptive_rate = min(0.99, adaptive_rate + 0.05)
            elif acceptance_rate > 0.8:
                # Speed up cooling if acceptance is too high
                adaptive_rate = max(0.7, adaptive_rate - 0.05)

            return temperature * adaptive_rate

        schedule = self.config.cooling_schedule

        if schedule == CoolingSchedule.EXPONENTIAL:
            return temperature * self.config.cooling_rate

        elif schedule == CoolingSchedule.LINEAR:
            linear_decrease = (
                self.config.initial_temp - self.config.min_temp
            ) / self.config.max_iterations
            return max(self.config.min_temp, temperature - linear_decrease)

        elif schedule == CoolingSchedule.LOGARITHMIC:
            return self.config.initial_temp / (1 + math.log(1 + iteration))

        elif schedule == CoolingSchedule.QUADRATIC:
            return self.config.initial_temp / (1 + iteration**2)

        else:
            # Default to exponential
            return temperature * self.config.cooling_rate

    def _should_restart(
        self, energy_history: List[float], recent_iterations: int = 100
    ) -> bool:
        """
        Determine if algorithm should restart based on stagnation.

        Args:
            energy_history: History of best energies
            recent_iterations: Number of recent iterations to check

        Returns:
            True if should restart, False otherwise
        """
        if len(energy_history) < recent_iterations:
            return False

        # Get recent energies
        recent_energies = energy_history[-recent_iterations:]

        # Calculate improvement ratio
        min_recent = min(recent_energies)
        max_recent = max(recent_energies)

        if max_recent - min_recent < 1e-10:  # Avoid division by zero
            return False

        improvement_ratio = (recent_energies[0] - min_recent) / (
            max_recent - min_recent
        )

        return improvement_ratio < self.config.restart_threshold

    def optimize(
        self,
        solution_generator: SolutionGenerator,
        initial_solution: Optional[Any] = None,
        verbose: bool = False,
        progress_callback: Optional[Callable[[int, float, float, Any], None]] = None,
    ) -> SimulatedAnnealingResult:
        """
        Execute the simulated annealing optimization.

        Args:
            solution_generator: Object implementing SolutionGenerator interface
            initial_solution: Initial solution (generated if None)
            verbose: Print progress information
            progress_callback: Callback function called each iteration
                Parameters: iteration, temperature, energy, solution

        Returns:
            SimulatedAnnealingResult with optimization results
        """
        start_time = time.time()

        # Initialize solution
        current_solution = initial_solution or solution_generator.generate_initial()
        current_energy = solution_generator.calculate_energy(current_solution)

        # Best solution tracking
        best_solution = current_solution
        best_energy = current_energy

        # Initialize temperature
        temperature = self.config.initial_temp

        # History tracking
        energies_history = [current_energy]
        temperatures_history = [temperature]

        # Acceptance tracking
        accepted_count = 0
        acceptance_rates = []

        # Main optimization loop
        iteration = 0
        while (
            temperature > self.config.min_temp
            and iteration < self.config.max_iterations
        ):
            # Generate neighbor solution
            neighbor_solution = solution_generator.generate_neighbor(
                current_solution, temperature
            )
            neighbor_energy = solution_generator.calculate_energy(neighbor_solution)

            # Calculate acceptance probability
            acceptance_prob = self._acceptance_probability(
                current_energy, neighbor_energy, temperature
            )

            # Decide whether to accept the neighbor
            if acceptance_prob > random.random():
                current_solution = neighbor_solution
                current_energy = neighbor_energy
                accepted_count += 1

                # Update statistics
                if neighbor_energy < current_energy:
                    self.stats["accepted_better"] += 1
                else:
                    self.stats["accepted_worse"] += 1

            # Update best solution
            if current_energy < best_energy:
                best_solution = current_solution
                best_energy = current_energy

            # Calculate acceptance rate (sliding window of 100 iterations)
            if iteration > 0 and iteration % 100 == 0:
                recent_accepted = (
                    sum(acceptance_rates[-100:]) if acceptance_rates else 0
                )
                acceptance_rate = recent_accepted / min(100, iteration)
            else:
                acceptance_rate = (
                    accepted_count / (iteration + 1) if iteration > 0 else 0
                )

            # Cool the temperature
            temperature = self._cool_temperature(
                temperature, iteration, acceptance_rate
            )

            # Check for restart
            if self._should_restart(energies_history):
                if verbose:
                    print(f"Iteration {iteration}: Restarting from best solution")
                current_solution = best_solution
                current_energy = best_energy
                temperature = (
                    self.config.initial_temp * 0.5
                )  # Restart with half initial temp
                self.stats["restarts"] += 1

            # Record history
            energies_history.append(best_energy)
            temperatures_history.append(temperature)
            acceptance_rates.append(1 if acceptance_prob > random.random() else 0)

            # Call progress callback
            if progress_callback:
                progress_callback(iteration, temperature, best_energy, best_solution)

            # Print progress if verbose
            if verbose and iteration % 500 == 0:
                print(
                    f"Iteration {iteration:6d}: "
                    f"Temp={temperature:8.4f}, "
                    f"Energy={best_energy:10.6f}, "
                    f"Accept Rate={acceptance_rate:5.3f}"
                )

            iteration += 1

        # Calculate final statistics
        execution_time = time.time() - start_time
        total_acceptance_rate = accepted_count / iteration if iteration > 0 else 0

        # Update statistics
        self.stats["total_iterations"] = iteration

        if verbose:
            self._print_summary(
                best_energy, execution_time, iteration, total_acceptance_rate
            )

        return SimulatedAnnealingResult(
            best_solution=best_solution,
            best_energy=best_energy,
            energies_history=energies_history,
            temperatures_history=temperatures_history,
            execution_time=execution_time,
            iterations=iteration,
            accepted_solutions=accepted_count,
            acceptance_rate=total_acceptance_rate,
        )

    def _print_summary(
        self,
        best_energy: float,
        execution_time: float,
        iterations: int,
        acceptance_rate: float,
    ) -> None:
        """Print optimization summary"""
        print("\n" + "=" * 60)
        print("SIMULATED ANNEALING OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Total iterations:      {iterations}")
        print(f"Execution time:        {execution_time:.3f} seconds")
        print(f"Best energy found:     {best_energy:.8f}")
        print(f"Acceptance rate:       {acceptance_rate:.2%}")
        print(f"Accepted better:       {self.stats['accepted_better']}")
        print(f"Accepted worse:        {self.stats['accepted_worse']}")
        print(f"Restarts performed:    {self.stats['restarts']}")
        print("=" * 60)

    def reset_statistics(self) -> None:
        """Reset algorithm statistics"""
        self.stats = {
            "total_iterations": 0,
            "accepted_worse": 0,
            "accepted_better": 0,
            "restarts": 0,
        }


@dataclass
class City:
    """City representation for TSP"""

    x: float
    y: float
    name: str
    id: int

    def distance_to(self, other: "City") -> float:
        """Calculate Euclidean distance to another city"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class TSPGenerator(SolutionGenerator):
    """Solution generator for Traveling Salesman Problem"""

    def __init__(self, cities: List[City]):
        self.cities = cities
        self.n_cities = len(cities)

    def generate_initial(self) -> List[City]:
        """Generate initial random route"""
        route = self.cities.copy()
        random.shuffle(route)
        return route

    def generate_neighbor(
        self, current_route: List[City], temperature: float
    ) -> List[City]:
        """Generate neighbor route using various mutation operators"""
        new_route = current_route.copy()

        # Choose mutation operator based on temperature
        # At high temperatures, use more disruptive mutations
        rand_val = random.random()

        if rand_val < 0.33:  # Swap two cities
            i, j = random.sample(range(self.n_cities), 2)
            new_route[i], new_route[j] = new_route[j], new_route[i]

        elif rand_val < 0.66:  # Reverse a segment
            i, j = sorted(random.sample(range(self.n_cities), 2))
            new_route[i : j + 1] = reversed(new_route[i : j + 1])

        else:  # Insert city at different position
            if self.n_cities > 3:
                city_idx = random.randint(0, self.n_cities - 1)
                insert_idx = random.randint(0, self.n_cities - 2)
                if insert_idx >= city_idx:
                    insert_idx += 1
                city = new_route.pop(city_idx)
                new_route.insert(insert_idx, city)

        return new_route

    def calculate_energy(self, route: List[City]) -> float:
        """Calculate total route distance (energy)"""
        total_distance = 0.0
        for i in range(self.n_cities):
            total_distance += route[i].distance_to(route[(i + 1) % self.n_cities])
        return total_distance


def create_random_cities(n_cities: int = 20, seed: Optional[int] = None) -> List[City]:
    """Create random cities for TSP"""
    if seed is not None:
        random.seed(seed)

    cities = []
    for i in range(n_cities):
        cities.append(
            City(
                x=random.uniform(0, 100),
                y=random.uniform(0, 100),
                name=f"City_{i}",
                id=i,
            )
        )
    return cities


def run_tsp_example():
    """Example: Traveling Salesman Problem optimization"""
    print("\n" + "=" * 60)
    print("TRAVELING SALESMAN PROBLEM - SIMULATED ANNEALING")
    print("=" * 60)

    # Create problem instance
    cities = create_random_cities(15, seed=42)
    print(f"Created {len(cities)} random cities")

    # Create solution generator
    tsp_generator = TSPGenerator(cities)

    # Create initial solution
    initial_route = tsp_generator.generate_initial()
    initial_distance = tsp_generator.calculate_energy(initial_route)
    print(f"Initial route distance: {initial_distance:.2f}")

    # Configure simulated annealing
    config = SAConfig(
        initial_temp=1000.0,
        min_temp=1e-7,
        cooling_rate=0.99,
        max_iterations=5000,
        cooling_schedule=CoolingSchedule.EXPONENTIAL,
        adaptive_cooling=True,
        restart_threshold=0.05,
    )

    # Create and run optimizer
    sa = SimulatedAnnealing(config)
    result = sa.optimize(
        solution_generator=tsp_generator, initial_solution=initial_route, verbose=True
    )

    # Print results
    print(f"\nOptimization completed!")
    print(f"Best distance: {result.best_energy:.2f}")
    print(
        f"Improvement: {((initial_distance - result.best_energy) / initial_distance * 100):.1f}%"
    )
    print(f"Route: {' -> '.join([city.name for city in result.best_solution])}")

    return result


class FunctionOptimizer(SolutionGenerator):
    """Solution generator for mathematical function optimization"""

    def __init__(
        self,
        function: Callable[[List[float]], float],
        bounds: List[Tuple[float, float]],
        dimension: int,
    ):
        self.function = function
        self.bounds = bounds
        self.dimension = dimension

    def generate_initial(self) -> List[float]:
        """Generate initial random solution within bounds"""
        return [random.uniform(b[0], b[1]) for b in self.bounds]

    def generate_neighbor(
        self, current_solution: List[float], temperature: float
    ) -> List[float]:
        """Generate neighbor solution with temperature-dependent variance"""
        new_solution = current_solution.copy()

        # Temperature-dependent mutation strength
        # Higher temperature -> larger mutations
        mutation_strength = 0.1 * temperature / 100

        for i in range(self.dimension):
            # Gaussian mutation with temperature-dependent variance
            mutation = random.gauss(0, mutation_strength)
            new_value = new_solution[i] + mutation

            # Apply bounds
            lower_bound, upper_bound = self.bounds[i]
            if new_value < lower_bound:
                new_value = lower_bound + abs(new_value - lower_bound) % (
                    upper_bound - lower_bound
                )
            elif new_value > upper_bound:
                new_value = upper_bound - abs(new_value - upper_bound) % (
                    upper_bound - lower_bound
                )

            new_solution[i] = new_value

        return new_solution

    def calculate_energy(self, solution: List[float]) -> float:
        """Calculate function value (energy)"""
        return self.function(solution)


def rastrigin_function(x: List[float]) -> float:
    """
    Rastrigin Function - Benchmark optimization function

    Global minimum: f(0,0,...,0) = 0
    Search domain: -5.12 <= x_i <= 5.12
    """
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * math.cos(2 * math.pi * xi)) for xi in x])


def run_function_optimization_example():
    """Example: Mathematical function optimization"""
    print("\n" + "=" * 60)
    print("MATHEMATICAL FUNCTION OPTIMIZATION - RASTRIGIN FUNCTION")
    print("=" * 60)

    # Problem setup
    dimension = 5
    bounds = [(-5.12, 5.12) for _ in range(dimension)]

    # Create solution generator
    func_generator = FunctionOptimizer(rastrigin_function, bounds, dimension)

    # Generate initial solution
    initial_solution = func_generator.generate_initial()
    initial_value = func_generator.calculate_energy(initial_solution)
    print(f"Dimension: {dimension}")
    print(f"Initial solution: {[round(x, 4) for x in initial_solution]}")
    print(f"Initial value: {initial_value:.4f}")

    # Configure simulated annealing
    config = SAConfig(
        initial_temp=50.0,
        min_temp=1e-10,
        cooling_rate=0.97,
        max_iterations=3000,
        cooling_schedule=CoolingSchedule.EXPONENTIAL,
        adaptive_cooling=True,
    )

    # Create and run optimizer
    sa = SimulatedAnnealing(config)
    result = sa.optimize(
        solution_generator=func_generator,
        initial_solution=initial_solution,
        verbose=True,
    )

    # Print results
    print(f"\nOptimization completed!")
    print(f"Best solution: {[round(x, 6) for x in result.best_solution]}")
    print(f"Best value: {result.best_energy:.10f}")
    print(f"Theoretical optimum: 0.0000000000")

    return result


try:
    import matplotlib.pyplot as plt

    def plot_optimization_results(
        result: SimulatedAnnealingResult, title: str = "Simulated Annealing Results"
    ):
        """Plot optimization progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Energy progression
        axes[0, 0].plot(result.energies_history, "b-", alpha=0.7, linewidth=1)
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Best Energy")
        axes[0, 0].set_title("Energy Progression")
        axes[0, 0].grid(True, alpha=0.3)

        # Temperature progression
        axes[0, 1].semilogy(result.temperatures_history, "r-", alpha=0.7, linewidth=1)
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Temperature (log scale)")
        axes[0, 1].set_title("Temperature Schedule")
        axes[0, 1].grid(True, alpha=0.3)

        # Energy vs Temperature
        axes[1, 0].scatter(
            result.temperatures_history,
            result.energies_history,
            c=range(len(result.energies_history)),
            cmap="viridis",
            alpha=0.5,
            s=1,
        )
        axes[1, 0].set_xlabel("Temperature")
        axes[1, 0].set_ylabel("Energy")
        axes[1, 0].set_title("Energy vs Temperature")
        axes[1, 0].set_xscale("log")
        axes[1, 0].grid(True, alpha=0.3)

        # Histogram of energy improvements
        energy_differences = [
            result.energies_history[i] - result.energies_history[i - 1]
            for i in range(1, len(result.energies_history))
        ]
        axes[1, 1].hist(energy_differences, bins=50, alpha=0.7, edgecolor="black")
        axes[1, 1].set_xlabel("Energy Improvement")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Energy Improvement Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

except ImportError:

    def plot_optimization_results(
        result: SimulatedAnnealingResult, title: str = "Simulated Annealing Results"
    ):
        print(
            "Matplotlib not available for plotting. Install with: pip install matplotlib"
        )


if __name__ == "__main__":
    print("SIMULATED ANNEALING ALGORITHM IMPLEMENTATION")
    print("=" * 60)

    # Run TSP example
    tsp_result = run_tsp_example()

    # Run function optimization example
    func_result = run_function_optimization_example()

    # Plot results if matplotlib is available
    try:
        plot_optimization_results(tsp_result, "TSP Optimization Results")
        plot_optimization_results(
            func_result, "Rastrigin Function Optimization Results"
        )
    except:
        pass
