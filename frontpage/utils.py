from rich.console import Console 
import itertools as it 

def batched(iterable, n=56):
    "Batch data into tuples of length n. The last batch may be shorter."
    if n < 1:
        raise ValueError('n must be at least one')
    iters = iter(iterable)
    while batch := tuple(it.islice(iters, n)):
        yield batch

console = Console()