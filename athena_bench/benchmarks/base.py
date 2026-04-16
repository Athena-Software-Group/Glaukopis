import abc

class Benchmark(abc.ABC):
    """Base class for CTI benchmarks."""
    
    def __init__(self, model_name, num_rows=None):
        self.model_name = model_name
        self.num_rows = num_rows
    
    @abc.abstractmethod
    def generate_responses(self, cleanup=False):
        """Generate and save model responses for the task.
        
        Args:
            cleanup (bool): Whether to clean up model from memory after each inference
        """
        pass