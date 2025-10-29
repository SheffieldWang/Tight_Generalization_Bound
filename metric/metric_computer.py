import jax.numpy as jnp

class MetricComputer:
    def __init__(self):
        self.metrics = {}

    def compute_accuracy(self,outputs,targets):
        # Check if logits sum to 1    
        predictions = jnp.sign(outputs)
        accuracy = jnp.sum(predictions == targets) / len(targets)
        return accuracy

    def compute_abs_error(self,predictions,targets):
        abs_error = jnp.mean(jnp.abs(predictions - targets))
        return abs_error



        