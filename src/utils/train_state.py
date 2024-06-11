import jax
import flax, optax
import flax.linen as nn
from flax import struct
from functools import partial
from typing import Any, Tuple, Optional, Callable, Union

nonpytree_field = partial(struct.field, pytree_node=False)      # which is not vectorizable
Params = flax.core.FrozenDict[str, Any]


class TrainState(struct.PyTreeNode):
    step: int
    apply_fn: Callable[..., Any] = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Params
    extra_variables: Optional[Params]
    tx: Optional[optax.GradientTransformation] = nonpytree_field()
    opt_state: Optional[optax.OptState]
    
    @classmethod
    def create(cls,
               model_def: nn.Module,
               params: Params,
               tx: Optional[optax.GradientTransformation] = None,
               extra_variables: Optional[Params] = None,
               **kwargs) -> "TrainState":
        
        opt_state = None
        if tx is not None:
            opt_state = tx.init(params)
            
        if extra_variables is None:
            extra_variables = flax.core.FrozenDict()
            
        return cls(step=0,
                   apply_fn=model_def.apply,
                   model_def=model_def,
                   params=params,
                   extra_variables=extra_variables,
                   tx=tx,
                   opt_state=opt_state,
                   **kwargs)
        
    def __call__(self,
                 *args,
                 params: Optional[Params] = None,
                 extra_variabels: Optional[Params] = None,
                 method: Union[str, Callable, None] = None,
                 **kwargs):
        
        if params is None:
            params = self.params
            
        if extra_variabels is None:
            extra_variabels = self.extra_variables
        variables = {"params": params, **extra_variabels}
        
        if isinstance(method, str):
            method = getattr(self.model_def, method)
        
        return self.apply_fn(variables, *args, method=method, **kwargs)
    
    def apply_gradients(self, grads, **kwargs) -> "TrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs
        )
        
    def apply_loss_fn(self, loss_fn, has_aux: bool = False, pmap_axis=None) -> Tuple["TrainState", Any]:
        if has_aux:
            grads, aux = jax.grad(loss_fn, has_aux=True)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                aux = jax.lax.pmean(aux, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads), aux
        else:
            grads = jax.grad(loss_fn)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads), None
        
        # has_aux = has_aux or mutable    # if mutable, then has_aux is True
        
        # out = jax.grad(loss_fn, has_aux=has_aux)(self.params)
        # grads, aux = out if has_aux else (out, None)
        # extra_variabels, info = aux if mutable else (aux, self.extra)
        
        # if pmap_axis is not None:
        #     grads = jax.lax.pmean(grads, axis_name=pmap_axis)
        #     if has_aux:
        #         info = jax.lax.pmean(info, axis_name=pmap_axis)
        
        # return self.apply_gradients(grads=grads, extra_variabels=extra_variabels), aux