from alive_progress import alive_bar
from dimod import SimulatedAnnealingSampler, ExactCQMSolver
from dwave.preprocessing.composites import FixVariablesComposite
from dwave.preprocessing.presolve import Presolver


def simple_simulated_annealing(bqm, invert, num_of_nodes, num_of_sol):
    with alive_bar(1) as bar:   
        print("No preprocessing")
        sampler = SimulatedAnnealingSampler()
        solution_bqm = sampler.sample(bqm)
        selected_medoids_bqm = [[i for i in range(num_of_nodes) if invert(sample)[f'z_{i}'] == 1] for sample in solution_bqm.samples()][:num_of_sol]
        bar()
    return selected_medoids_bqm


def roof_duality_simulated_annealing(bqm, num_of_nodes, num_of_sol):
    with alive_bar(1) as bar:
        print("Fixed roof duality")
        sampler_fixed = FixVariablesComposite(SimulatedAnnealingSampler(), algorithm='roof_duality')
        solution_bqm_fixed = sampler_fixed.sample(bqm)
        selected_medoids_bqm_fixed = [[i for i in range(num_of_nodes) if sample[f'z_{i}'] == 1] for sample in solution_bqm_fixed.samples()][:num_of_sol]
        bar()
    return selected_medoids_bqm_fixed


def exact_cqm_solve(cqm, num_of_nodes, num_of_sol):
    with alive_bar(1) as bar:
        print("Exact CQM solver")
        sampler = ExactCQMSolver() 
        result = sampler.sample_cqm(cqm)
        feasible_samples = result.filter(lambda d: d.is_feasible)
        selected_medoids = [[i for i in range(num_of_nodes) if sample[f'z_{i}'] == 1] for sample in feasible_samples.samples()[:num_of_sol]]
        bar()
    return selected_medoids