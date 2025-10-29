import jax.numpy as jnp
import optax
import chex


def hinge_loss_fn(model,data,target):
    outputs = model(data)
    loss = jnp.mean(optax.hinge_loss(outputs,target))
    return loss,outputs



def hinge_loss_with_epsilon(
    predictor_outputs: chex.Array, targets: chex.Array, epsilon: float = float('inf')
) -> chex.Array:
  """Computes the hinge loss for binary classification.

  Args:
    predictor_outputs: Outputs of the decision function.
    targets: Target values. Target values should be strictly in the set {-1, 1}.
    epsilon: Maximum value for the loss. Default is infinity (no clipping).

  Returns:
    loss value.
  """
  return jnp.maximum(0, 1 - predictor_outputs * targets)


def hinge_loss_fn_with_epsilon_fn(model,data,target,epsilon=1):
    outputs = model(data)
    loss = jnp.mean(hinge_loss_with_epsilon(outputs,target,epsilon))
    return loss,outputs



def mse_loss_fn(model,data,target):
    outputs = model(data)
    loss = jnp.mean((outputs-target)**2)
    return loss,outputs

# def cross_entropy_loss_fn(model,data,target):
#     logits = model(data)
#     loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=target).mean()
#     return loss,logits