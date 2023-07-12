
def _base_file_name(algo, n_terms: int = 3, membership_fn: str = 'trapezoid', t_conorm: str = 'max', t_norm: str='min'):
    return f'{algo}_{n_terms}_terms_{membership_fn}_membership_{t_norm}_t_norm_{t_conorm}_t_conorm'