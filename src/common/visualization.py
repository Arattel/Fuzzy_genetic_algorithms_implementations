from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def visualize_efga(history: pd.DataFrame, windowsize_best: int = 10, windowsize_avg: int = 10, function_name: str = ''):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    for seed in history['seed'].unique():
        ax1.plot(np.arange((history['seed'] ==  seed).sum()), history[history['seed'] ==  seed].avg_fitness.rolling(windowsize_avg).mean())
    
    for seed in history['seed'].unique():
        ax2.plot(np.arange((history['seed'] ==  seed).sum()), history[history['seed'] ==  seed].best_fitness.rolling(windowsize_best).mean())
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('Fitness')
    ax2.set_ylabel('Fitness')
    ax1.set_title(f'Mean fitness of population, noving average of {windowsize_avg}')
    ax2.set_title(f'Best fitness of population, noving average of {windowsize_best}')
    fig.suptitle(f'Optimization of {function_name}')
    # plt.show()
    return fig



def visualize_gendered(history: pd.DataFrame, windowsize_best: int = 10, windowsize_avg: int = 10, windowsize_pop=5,  function_name: str = ''):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figheight(7)
    fig.set_figwidth(17)
    for seed in history['seed'].unique():
        ax1.plot(np.arange((history['seed'] ==  seed).sum()), history[history['seed'] ==  seed].avg_fitness.rolling(windowsize_avg).mean())
    
    for seed in history['seed'].unique():
        ax2.plot(np.arange((history['seed'] ==  seed).sum()), history[history['seed'] ==  seed].best_fitness.rolling(windowsize_best).mean())

    for seed in history['seed'].unique():
        print('wqehwe')
        ax3.plot(np.arange((history['seed'] ==  seed).sum()), history[history['seed'] ==  seed].population_size.rolling(windowsize_pop).mean())
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax3.set_xlabel('Epoch')
    ax1.set_ylabel('Fitness')
    ax2.set_ylabel('Fitness')
    ax3.set_xlabel('Population size')
    ax1.set_title(f'Mean fitness of population, moving average of {windowsize_avg}')
    ax2.set_title(f'Best fitness of population, moving average of {windowsize_best}')
    ax3.set_title(f'Population, moving average of {windowsize_pop}')

    fig.suptitle(f'Optimization of {function_name}')
    return fig

    # plt.show()