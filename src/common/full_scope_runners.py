from datetime import datetime
import os
import warnings
import json


from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm


from .experiment_runners import run_experiments_efga, run_experiments_gendered
from .visualization import visualize_efga, visualize_gendered
def _get_ts():
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    return ts

def experiments_efga(func, n_experiments=5, epochs=500, N=500, n_terms_priority=6, n_terms_params=3, log_dir: str = 'log', 
                     plot_dir: str = 'img', population_scale=500, mutation_scale=.5,):
    fn_name = func.__name__
    current_timestamp = _get_ts()

    # File outputs
    output_params = f'{fn_name}_efga_params_{current_timestamp}.json'
    output_params = os.path.join(log_dir, output_params)
    output_experiment_logs = f'{fn_name}_experiment_efga_logs_{current_timestamp}.csv'
    output_experiment_logs = os.path.join(log_dir, output_experiment_logs)
    output_plot = f'{fn_name}_efga_summary_plot_{current_timestamp}.png'
    output_plot = os.path.join(plot_dir, output_plot)

    experiment_logs, params = run_experiments_efga(n_experiments=n_experiments, 
                                                   epochs=epochs, N=N, n_terms_priority=n_terms_priority,
                                                    n_terms_params=n_terms_params, population_scale=population_scale, mutation_scale=mutation_scale, 
                                                    fitness_fn=func)
    
    with open(output_params, 'w') as f:
        json.dump(params, f)
        
    experiment_logs.to_csv(output_experiment_logs)
    figure = visualize_efga(experiment_logs, windowsize_best=20, function_name=fn_name)
    figure.savefig(output_plot)

def experiments_gendered(func, n_experiments=5, epochs=500, N=500, n_partitions=3, log_dir: str = 'log', 
                     plot_dir: str = 'img',  population_scale=500, mutation_scale=.5):
    fn_name = func.__name__
    current_timestamp = _get_ts()

    # File outputs
    output_params = f'{fn_name}_efga_params_{current_timestamp}.json'
    output_params = os.path.join(log_dir, output_params)
    output_experiment_logs = f'{fn_name}_experiment_efga_logs_{current_timestamp}.csv'
    output_experiment_logs = os.path.join(log_dir, output_experiment_logs)
    output_plot = f'{fn_name}_gendered_summary_plot_{current_timestamp}.png'
    output_plot = os.path.join(plot_dir, output_plot)

    experiment_logs, params = run_experiments_gendered(n_experiments=n_experiments, 
                                                   epochs=epochs, N=N,
                                                    n_partitions=n_partitions,  population_scale=population_scale, 
                                                    mutation_scale=mutation_scale, fitness_fn=func)
    
    with open(output_params, 'w') as f:
        json.dump(params, f)
        
    experiment_logs.to_csv(output_experiment_logs)
    figure = visualize_gendered(experiment_logs, windowsize_best=10, function_name=fn_name)
    figure.savefig(output_plot)