# Example of TypeVar with bound 
# it ensures type strictness at runtime

from typing import TypeVar

Numeric = TypeVar('Numeric', bound=int | float)  # Only int or float

def add(a: Numeric, b: Numeric) -> Numeric:
    return a + b

print(add(5, 10))      # ✅ Valid
print(add(3.14, 2.71)) # ✅ Valid
print(add("a", "b"))   # ❌ Error
