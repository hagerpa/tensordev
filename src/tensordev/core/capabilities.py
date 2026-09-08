"""Private capability tokens shared by concrete tensor cores."""

from __future__ import annotations


# Identity, rather than a boolean or string, makes the optimized route an
# explicit contract with the built-in core classes.  A compatible extension
# may opt in deliberately by importing the private token and defining the
# class attribute itself.
_WORDWISE_SIGNATURE_PROTOCOL = object()


__all__ = []
