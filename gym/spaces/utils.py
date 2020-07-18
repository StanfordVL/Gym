import numpy as np

from . import Box
from . import Discrete
from . import MultiDiscrete
from . import MultiBinary
from . import Tuple
from . import Dict


def flatdim(space):
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return int(space.n)
    elif isinstance(space, Tuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return int(np.prod(space.shape))
    else:
        raise NotImplementedError


def flatten(space, x):
    if isinstance(space, Box):
        return np.asarray(x, dtype=np.float32).flatten()
    elif isinstance(space, Discrete):
        onehot = np.zeros(space.n, dtype=np.float32)
        onehot[x] = 1.0
        return onehot
    elif isinstance(space, Tuple):
        return np.concatenate([flatten(s, x_part) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, Dict):
        return np.concatenate([flatten(s, x[key]) for key, s in space.spaces.items()])
    elif isinstance(space, MultiBinary):
        return np.asarray(x).flatten()
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x).flatten()
    else:
        raise NotImplementedError


def unflatten(space, x):
    if isinstance(space, Box):
        return np.asarray(x, dtype=np.float32).reshape(space.shape)
    elif isinstance(space, Discrete):
        return int(np.nonzero(x)[0][0])
    elif isinstance(space, Tuple):
        dims = [flatdim(s) for s in space.spaces]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [unflatten(s, flattened) 
                            for flattened, s in zip(list_flattened, space.spaces)]
        return tuple(list_unflattened)
    elif isinstance(space, Dict):
        dims = [flatdim(s) for s in space.spaces.values()]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [(key, unflatten(s, flattened)) 
                            for flattened, (key, s) in zip(list_flattened, space.spaces.items())]
        return dict(list_unflattened)
    elif isinstance(space, MultiBinary):
        return np.asarray(x).reshape(space.shape)
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x).reshape(space.shape)
    else:
        raise NotImplementedError
