# RED Phase: custom-autograd-functions Baseline Testing

**Skill**: `yzmir/pytorch-engineering/custom-autograd-functions`
**Phase**: RED (Baseline Without Skill)
**Date**: 2025-10-30

## Purpose

Establish baseline behavior for agents implementing custom autograd functions WITHOUT expert guidance. This documents the mistakes, rationalizations, and failure modes that occur when agents lack deep understanding of torch.autograd.Function.

## Test Scenarios

### Scenario 1: Custom Activation Function Without Function Class

**Context**: User needs a custom activation function (Swish-like variant) not available in PyTorch.

**User Request**:
> "I need to implement a custom activation function: `f(x) = x * sigmoid(beta * x)` where beta is learnable. The gradient should be computed correctly for backprop. Implement this quickly."

**Expected Baseline Behavior (Without Skill)**:
- Agent writes normal Python function instead of torch.autograd.Function
- Wraps in nn.Module but doesn't use Function.apply
- Assumes PyTorch autograd "will figure it out"
- No gradient verification with gradcheck
- May work for simple cases but breaks with complex operations

**Baseline Implementation**:
```python
import torch
import torch.nn as nn

class CustomActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Just a regular function - no custom autograd
        return x * torch.sigmoid(self.beta * x)
```

**Why This Fails**:
- ✅ Autograd actually works here (composed from standard ops)
- ❌ But agent doesn't understand WHEN custom Function is needed
- ❌ No verification that gradient is correct
- ❌ If operation was truly custom (e.g., from C++ extension), this would fail silently

**Rationalization**:
> "PyTorch's autograd will automatically compute gradients through sigmoid and multiplication. No need for custom Function class - that's only for advanced use cases."

---

### Scenario 2: Incorrect Backward Implementation

**Context**: User implements custom matrix operation with manual gradient computation.

**User Request**:
> "I need a custom linear transformation with a special gradient behavior: `forward: Y = X @ W`, but in backward, clip the gradient to W between -1 and 1. Implement the custom autograd function."

**Expected Baseline Behavior (Without Skill)**:
- Agent attempts torch.autograd.Function
- Implements forward() correctly
- Implements backward() but with gradient bugs:
  - Wrong gradient formula (transposes incorrect)
  - Shape mismatches
  - Doesn't return gradient for all inputs
  - Clips wrong gradient
- No gradcheck verification
- Gradient silently wrong

**Baseline Implementation**:
```python
import torch
from torch.autograd import Function

class CustomLinear(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return input @ weight

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        # BUG 1: Wrong gradient computation (no transpose)
        grad_input = grad_output @ weight  # Should be weight.t()

        # BUG 2: Wrong gradient for weight
        grad_weight = input @ grad_output  # Wrong order

        # BUG 3: Clips wrong gradient
        grad_input = torch.clamp(grad_input, -1, 1)  # Should clip weight grad

        return grad_input, grad_weight

# BUG 4: No gradcheck verification
custom_linear = CustomLinear.apply
```

**Why This Fails**:
- ❌ Gradient computation mathematically wrong
- ❌ Shape mismatches cause runtime errors
- ❌ No gradcheck to catch bugs
- ❌ Clips wrong gradient (requirement not met)

**Actual Failures**:
1. Runtime error: matrix dimension mismatch
2. If shapes happen to align, gradient silently incorrect
3. Training diverges or produces wrong results
4. Extremely hard to debug without gradcheck

**Rationalization**:
> "The gradient formula looks right - chain rule says multiply by grad_output. I tested it and shapes work. Don't need gradcheck because the math is straightforward."

---

### Scenario 3: Context (ctx) Misuse

**Context**: User implements custom convolution wrapper that needs to save multiple tensors for backward.

**User Request**:
> "Implement a custom autograd function that wraps torch.nn.functional.conv2d but logs statistics during forward and backward. Save input, weight, and stride for backward pass."

**Expected Baseline Behavior (Without Skill)**:
- Agent doesn't understand ctx.save_for_backward() rules
- Saves tensors incorrectly (as attributes, not save_for_backward)
- Mixes tensor and non-tensor data incorrectly
- Accesses saved_tensors in forward (crashes)
- Modifies saved tensors (breaks gradient)

**Baseline Implementation**:
```python
import torch
from torch.autograd import Function
import torch.nn.functional as F

class CustomConv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, stride):
        # BUG 1: Saves tensor as attribute (wrong)
        ctx.input = input  # Should use save_for_backward
        ctx.weight = weight  # Should use save_for_backward

        # BUG 2: Tries to save non-tensor with save_for_backward
        ctx.save_for_backward(stride)  # stride is int, not tensor

        output = F.conv2d(input, weight, stride=stride)

        # BUG 3: Tries to access saved_tensors in forward
        print(f"Saved input: {ctx.saved_tensors}")  # Crashes

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # BUG 4: Wrong unpacking (stride was saved incorrectly)
        input = ctx.input  # Works but wrong pattern
        weight = ctx.weight
        stride, = ctx.saved_tensors  # stride is not a tensor

        # BUG 5: Modifies saved tensor (breaks gradient computation)
        input = input * 2  # Modifying saved tensor!

        grad_input = F.conv2d(grad_output, weight, stride=stride)
        grad_weight = F.conv2d(input, grad_output, stride=stride)

        return grad_input, grad_weight, None
```

**Why This Fails**:
- ❌ ctx attribute saving doesn't track memory correctly
- ❌ Can't save non-tensors with save_for_backward
- ❌ saved_tensors only available in backward
- ❌ Modifying saved tensors breaks gradient graph
- ❌ Incorrect unpacking causes crashes

**Actual Failures**:
1. Runtime error when accessing saved_tensors in forward
2. Type error when trying to save int with save_for_backward
3. Gradient computation uses modified tensors (incorrect results)
4. Memory tracking broken (potential memory leaks)

**Rationalization**:
> "ctx is just a container to pass data from forward to backward. I can save things however I want. The implementation details don't matter as long as I can access the data."

---

### Scenario 4: In-Place Operation Breaking Autograd

**Context**: User implements custom ReLU with in-place modification for "efficiency".

**User Request**:
> "Implement a custom ReLU using torch.autograd.Function. Make it memory-efficient by using in-place operations."

**Expected Baseline Behavior (Without Skill)**:
- Agent implements Function with in-place operations
- Uses .data or direct tensor modification
- Doesn't understand in-place breaks autograd graph
- No warning about in-place dangers
- Silent failure or cryptic autograd errors

**Baseline Implementation**:
```python
import torch
from torch.autograd import Function

class CustomReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        # BUG 1: In-place operation in forward
        output = input.clone()
        output[output < 0] = 0  # This is fine, but agent might do:
        # input[input < 0] = 0  # IN-PLACE - breaks autograd!
        # return input

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # BUG 2: In-place operation in backward
        grad_output[input < 0] = 0  # BREAKS AUTOGRAD!
        return grad_output

# Usage breaks in complex graph
custom_relu = CustomReLU.apply
```

**Why This Fails**:
- ❌ In-place operations modify tensors that other operations depend on
- ❌ Breaks autograd graph silently
- ❌ Error message cryptic: "modified by an inplace operation"
- ❌ Hard to debug because error appears far from source

**Actual Failures**:
1. RuntimeError: "one of the variables needed for gradient computation has been modified by an inplace operation"
2. Gradient computation produces wrong results
3. Works in simple cases but breaks in complex graphs
4. Extremely confusing debugging experience

**Rationalization**:
> "In-place operations are more memory-efficient. I'm just modifying the output, not the input, so it should be fine. PyTorch should handle this."

---

## Summary of Baseline Failures

### Common Mistakes

1. **Not using Function class when needed**
   - Writing normal functions for custom ops
   - Assuming autograd "just works" for everything
   - No understanding of when Function is required

2. **Incorrect gradient computation**
   - Wrong mathematical formula (transpose errors, wrong order)
   - Shape mismatches (gradient shape ≠ input shape)
   - Not returning gradient for all inputs
   - Forgetting None for optional parameters

3. **Context (ctx) misuse**
   - Saving tensors as attributes instead of save_for_backward()
   - Trying to save non-tensors with save_for_backward()
   - Accessing saved_tensors in forward
   - Modifying saved tensors in backward

4. **No gradient verification**
   - Never running gradcheck
   - "Gradient looks right" based on inspection
   - No numerical verification
   - Bugs go unnoticed until training fails

5. **In-place operations**
   - Using in-place ops for "efficiency"
   - Not understanding they break autograd
   - Modifying grad_output or saved tensors directly

6. **Missing needs_input_grad checks**
   - Computing gradients even when not needed
   - Wasting computation
   - Not understanding the optimization

### Missing Knowledge

**Agents without this skill don't know**:
- When to use torch.autograd.Function vs normal functions
- How to properly implement forward() and backward()
- ctx.save_for_backward() is mandatory for tensors
- gradcheck is mandatory for verification
- In-place operations break autograd
- Gradient must match input shape exactly
- needs_input_grad optimization pattern
- How to debug gradient computation bugs

### Dangerous Rationalizations

| Rationalization | Why It's Wrong | Consequence |
|----------------|----------------|-------------|
| "Autograd will handle it" | Only works for standard ops | Silent failures for custom ops |
| "Gradient looks mathematically correct" | Implementation bugs invisible | Wrong training behavior |
| "Don't need gradcheck, I tested it" | Manual tests miss edge cases | Subtle gradient bugs |
| "In-place is more efficient" | Breaks autograd graph | Runtime errors, wrong results |
| "ctx details don't matter" | Memory tracking breaks | Memory leaks, crashes |
| "Shape will work out" | Must match exactly | Runtime errors |
| "Too complex for me" | Pattern is standardized | Avoids necessary tool |
| "Skip needs_input_grad check" | Wastes computation | Slow backward pass |

## What's Needed

A skill that:
1. ✅ Teaches when custom Function is truly needed
2. ✅ Provides complete Function template (forward/backward)
3. ✅ Makes ctx rules crystal clear (save_for_backward, saved_tensors)
4. ✅ MANDATES gradcheck verification (no exceptions)
5. ✅ Explains gradient computation patterns (matmul, broadcast, reduction)
6. ✅ Lists all common pitfalls (in-place, shapes, ctx misuse)
7. ✅ Provides debugging workflow (forward → gradcheck → shapes → magnitudes)
8. ✅ Shows memory efficiency patterns (checkpointing, selective saving)
9. ✅ Includes advanced patterns (double backward, custom hooks)
10. ✅ Strong rationalization counters (make gradcheck non-negotiable)

## Next Steps

Proceed to GREEN phase:
- Implement comprehensive custom-autograd-functions skill
- Address all baseline failures
- Make gradcheck mandatory
- Provide complete Function patterns
- Include extensive gradient computation examples
- Build strong rationalization resistance
