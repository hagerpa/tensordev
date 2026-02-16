# Framework-flexible Tensor Algebra: Signatures & Shuffles

## Status quo
This package is currently in development. You can use it by creating core objects. Here are a few examples:

```python
from tensordev import Universal
import numpy as np

NumpyCore = Universal(np.array(0).__array_namespace__)
# or in this case equivalently:
NumpyCore = Universal(np)
```

Now you can access tensor methods from `NumpyCore.tensor*`.  
For instance, let’s create some Brownian sample paths for later reuse:

```python
B, L, d, depth = 1000, 101, 3, 7
path_a = np.concatenate(
    [
        np.zeros((B, 1, d)),
        np.cumsum(
            np.array(np.random.randn(B, L - 1, d)) * (1.0 / (L - 1)) ** 2,
            axis=1,
        ),
    ],
    axis=1,
)
```

Compute the signature:

```python
sig_a = NumpyCore.tensor_path_signature(path_a, trunc=depth)
len(sig_a), [lvl.shape for lvl in sig_a]
```

```
>> [14s 205ms] (8,
 [(1000, 1),
  (1000, 3),
  (1000, 9),
  (1000, 27),
  (1000, 81),
  (1000, 243),
  (1000, 729),
  (1000, 2187)])
```

Create a second path, concatenate them, and check that Chen’s identity holds:

```python
path_b = np.array(np.random.randn(B, L - 1, d), dtype=np.float64) * (1.0 / (L - 1)) ** 2
path_b = np.concatenate([path_a[:, -1:, :], path_b], axis=1)
path_b = np.cumsum(path_b, axis=1)
path_c = np.concatenate([path_a, path_b[:, 1:]], axis=1)

sig_b = NumpyCore.tensor_development([path_b], trunc=depth)
sig_c = NumpyCore.tensor_development([path_c], trunc=depth, block_size=100)

sig_c_ = NumpyCore.tensor_product(sig_a, sig_b, trunc=depth)

for i in range(depth):
    assert np.allclose(sig_a[i], sig_c[i][:, 0])  # first block equals sig_a
    assert np.allclose(sig_c_[i], sig_c[i][:, 1])  # second block equals sig_a ⊗ sig_b
```

If you want framework-optimized code, import the corresponding core, e.g., for JAX:

```python
from tensordev import Jax
JaxCore = Jax()

sig_a = JaxCore.tensor_path_signature(path_a, trunc=depth)
```

```
>>> [265ms]
```

## Future
Later, this package will be installable and all methods will be available via:

```python
from tensordev import *
```

This will select the best available framework automatically on your system.  
The general entry point will also support zeroth levels, e.g.:

```python
tensor_product([None, path_a], [None, path_b])
```

```
>>> [None, None, path_a ⊗ path_b]
```

For framework-optimized code, you will also be able to import directly, e.g.:

```python
from tensordev.jax import *
```


