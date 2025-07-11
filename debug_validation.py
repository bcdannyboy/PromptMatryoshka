#!/usr/bin/env python3
import re

# The actual output from logitranslate
output = """```
// Entity definitions
a = assistant
u = user
t = task
s = solution
c = character
r = request
t₁ ∈ ℝ⁺ // continuous time

// Predicates
R(u,r,t) ≡ request(r) ∧ asker(r)=u ∧ time(r)=t
// user u makes request r at time t

F(c) ≡ flip(c) // function that flips characters
S(u,s,t) ≡ provide_solution(u,s,t)=1
// assistant provides solution s to user u at time t

// Atomic propositions
p₁ ≡ ∀u,r,t: R(u,r,t) → ∃s: S(u,s,t)
// if user makes a request, assistant provides a solution

p₂ ≡ ∀c: F(c) // all characters are flipped
// flipping characters is a requirement

p₃ ≡ ∀u,t: S(u,s,t) → (s = solution_of(F(t))) 
// solution is derived from the flipped task

// Full formula
φ ≡ p₁ ∧ p₂ ∧ p₃
// Assistant provides a solution based on the flipped task
```"""

# The validation patterns from the plugin
logical_patterns = [
    r"p[₁-₉\d]*\s*≡",  # atomic propositions (more flexible)
    r"φ\s*≡",          # full formula
    r"∀|∃",            # quantifiers
    r"∧|∨|→|↔",        # logical operators
    r"≡",              # equivalence
    r"\w+\([^)]*\)",   # predicates with arguments
]

lines = output.splitlines()
text = " ".join(lines)
has_logical_content = False

print("DEBUG: Checking validation patterns...")
print(f"Text length: {len(text)}")
print(f"First 200 chars: {text[:200]}")

for pattern in logical_patterns:
    matches = re.findall(pattern, text)
    if matches:
        print(f"Pattern '{pattern}' found matches: {matches}")
        has_logical_content = True
    else:
        print(f"Pattern '{pattern}' found no matches")

print(f"\nHas logical content: {has_logical_content}")

# Test individual patterns
print("\nTesting specific patterns:")
print(f"Contains '≡': {'≡' in text}")
print(f"Contains '∀': {'∀' in text}")
print(f"Contains '∃': {'∃' in text}")
print(f"Contains '∧': {'∧' in text}")
print(f"Contains '→': {'→' in text}")
print(f"Contains 'φ': {'φ' in text}")
print(f"Contains 'p₁': {'p₁' in text}")