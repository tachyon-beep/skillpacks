# Custom Autograd Functions in PyTorch

---
name: custom-autograd-functions
description: Master torch.autograd.Function for custom differentiable operations with correct gradients
pack: pytorch-engineering
author: yzmir-ai-engineering
version: 1.0.0
dependencies: [python, pytorch]
tags: [autograd, gradients, backpropagation, custom-operations, gradient-checking]
applies_to: [feature-implementation, performance-optimization, custom-operations]
---

## Overview

This skill teaches you to implement custom autograd functions in PyTorch using `torch.autograd.Function`. You'll learn when custom functions are necessary, how to implement forward and backward passes correctly, mandatory numerical verification with gradcheck, and common pitfalls that break gradient computation.

## When This Skill Applies

Use `torch.autograd.Function` when you encounter these situations:

### Symptoms That Require Custom Functions

1. **Custom operations not in PyTorch**: Implementing novel mathematical operations (special activations, custom loss components, domain-specific transformations)

2. **Wrapping external code**: Interfacing with C++/CUDA kernels, third-party libraries, or compiled extensions that PyTorch doesn't know about

3. **Custom gradient behavior**: Need non-standard gradient computation (gradient clipping in backward, sparsification, quantization, gradient routing)

4. **Memory optimization**: Implementing gradient checkpointing, fused operations, or selective materialization to reduce memory footprint

5. **Interfacing with non-differentiable code**: Wrapping operations that aren't naturally differentiable but have known gradient behavior

### When NOT to Use Custom Functions

Don't use `torch.autograd.Function` when:

- ❌ **Operation composable from existing ops**: If you can write it with standard PyTorch operations, autograd handles gradients automatically
- ❌ **Simple function wrapping**: Just wrapping `torch.nn.functional` operations gains nothing
- ❌ **Standard gradient computation**: No custom behavior needed - use regular PyTorch
- ❌ **Avoiding learning curve**: Custom functions aren't "advanced only" - use when appropriate

**Example**:
```python
# DON'T: Unnecessary custom function
class MyAdd(Function):  # ❌ Pointless wrapper
    @staticmethod
    def forward(ctx, a, b):
        return a + b
    @staticmethod
    def backward(ctx, grad):
        return grad, grad

# DO: Use PyTorch's autograd
output = a + b  # ✅ Autograd handles this correctly

# DON'T: Reimplement existing operations
class MyReLU(Function):  # ❌ Use torch.nn.functional.relu
    ...

# DO: Use built-in operations
output = torch.relu(input)  # ✅ Efficient and correct
```

## Core Pattern: Complete Function Implementation

### Basic Template

Every custom autograd function follows this pattern:

```python
import torch
from torch.autograd import Function

class MyCustomFunction(Function):
    """
    Custom autograd function template.

    Implements forward pass (computation) and backward pass (gradient).
    Context object (ctx) saves data between forward and backward.
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        """
        Forward pass: compute output from inputs.

        Args:
            ctx: Context object for saving tensors/data
            input: First input tensor
            weight: Second input tensor
            bias: Optional bias tensor

        Returns:
            output: Result tensor
        """
        # Save tensors needed for backward pass
        ctx.save_for_backward(input, weight, bias)

        # Save non-tensor data as ctx attributes
        # ctx.some_value = some_non_tensor_data

        # Compute forward pass
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients using chain rule.

        Args:
            ctx: Context object with saved data
            grad_output: Gradient of loss w.r.t. output (dL/dY)

        Returns:
            Tuple of gradients for each input to forward():
            (grad_input, grad_weight, grad_bias)
            Must return one gradient per forward() argument (None if not needed)
        """
        # Retrieve saved tensors
        input, weight, bias = ctx.saved_tensors

        # Initialize gradients to None
        grad_input = grad_weight = grad_bias = None

        # Compute gradients only if needed (efficiency)
        if ctx.needs_input_grad[0]:  # Check if input needs gradient
            grad_input = grad_output.mm(weight)  # dL/dX = dL/dY @ W

        if ctx.needs_input_grad[1]:  # Check if weight needs gradient
            grad_weight = grad_output.t().mm(input)  # dL/dW = dL/dY.T @ X

        if bias is not None and ctx.needs_input_grad[2]:  # Check if bias needs gradient
            grad_bias = grad_output.sum(0)  # Sum over batch dimension

        # Must return gradient for each forward() input (including ctx)
        # First return is always None (ctx doesn't need gradient)
        # But since ctx is implicit, return one per actual argument
        return grad_input, grad_weight, grad_bias

# Use the custom function
my_func = MyCustomFunction.apply  # Get callable
output = my_func(input_tensor, weight_tensor, bias_tensor)
loss = output.sum()
loss.backward()  # Calls MyCustomFunction.backward()
```

### Critical Rules

**Rule 1: Return gradient for EACH forward input**
```python
# forward signature: forward(ctx, a, b, c, d=None)
# backward must return: (grad_a, grad_b, grad_c, grad_d)

# ✅ CORRECT: 4 inputs → 4 gradient returns
def backward(ctx, grad_output):
    return grad_a, grad_b, grad_c, grad_d

# ❌ WRONG: Missing gradients
def backward(ctx, grad_output):
    return grad_a, grad_b  # Only 2 - will crash!

# ✅ CORRECT: Use None for unused gradients
def backward(ctx, grad_output):
    return grad_a, None, grad_c, None  # b and d don't need gradients
```

**Rule 2: Gradient shape must match input shape exactly**
```python
# If input.shape = (32, 128)
# Then grad_input.shape MUST BE (32, 128)

# ✅ CORRECT: Shapes match
assert grad_input.shape == input.shape

# ❌ WRONG: Shape mismatch causes runtime error
grad_input = some_computation()  # Shape (32, 64) - WRONG!
```

**Rule 3: Check needs_input_grad before computing**
```python
# Efficiency optimization: skip gradient computation if not needed

# ✅ CORRECT: Check before computing
if ctx.needs_input_grad[0]:
    grad_input = expensive_gradient_computation(...)
else:
    grad_input = None

# ❌ WASTEFUL: Always compute (slow)
grad_input = expensive_gradient_computation(...)  # Even if not needed
```

## Context Object (ctx) Rules

The context object `ctx` is how you pass data from forward to backward. Use it correctly or break everything.

### Rule 1: Use save_for_backward() for Tensors ONLY

```python
# ✅ CORRECT: Save tensors with save_for_backward()
@staticmethod
def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)  # Both are tensors
    return input @ weight

# ❌ WRONG: Saving tensors as attributes
@staticmethod
def forward(ctx, input, weight):
    ctx.input = input  # Breaks memory tracking
    ctx.weight = weight  # Breaks memory tracking
    return input @ weight

# Why it matters: save_for_backward() properly tracks tensor versions
# and memory. Attribute assignment doesn't, leading to bugs/crashes.
```

### Rule 2: Save Non-Tensor Data as Attributes

```python
# ✅ CORRECT: Non-tensor data as attributes
@staticmethod
def forward(ctx, input, kernel_size, stride):
    ctx.save_for_backward(input)  # Tensor
    ctx.kernel_size = kernel_size  # Integer - use attribute
    ctx.stride = stride  # Integer - use attribute
    return some_operation(input, kernel_size, stride)

# ❌ WRONG: Trying to save non-tensors with save_for_backward()
@staticmethod
def forward(ctx, input, kernel_size, stride):
    ctx.save_for_backward(input, kernel_size, stride)  # TypeError!
    # kernel_size and stride are ints, not tensors
```

### Rule 3: Access saved_tensors Only in Backward

```python
# ✅ CORRECT: Access in backward
@staticmethod
def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors  # Available here
    grad_input = grad_output @ weight.t()
    return grad_input, None

# ❌ WRONG: Access in forward
@staticmethod
def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)
    output = input @ weight
    saved = ctx.saved_tensors  # AttributeError! Not available in forward
    return output
```

### Rule 4: Never Modify Saved Tensors

```python
# ❌ WRONG: Modifying saved tensor
@staticmethod
def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    input = input * 2  # Creates new tensor - OK
    input *= 2  # IN-PLACE modification - BREAKS AUTOGRAD!
    grad_input = compute_gradient(input, grad_output)
    return grad_input

# ✅ CORRECT: Don't modify, or clone first
@staticmethod
def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    input_scaled = input * 2  # New tensor - safe
    grad_input = compute_gradient(input_scaled, grad_output)
    return grad_input
```

### Complete ctx Example

```python
class CompleteCtxExample(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride=1, training=True):
        # Save tensors with save_for_backward()
        ctx.save_for_backward(input, weight, bias)

        # Save non-tensor data as attributes
        ctx.stride = stride  # int
        ctx.training = training  # bool

        # Compute forward
        output = some_computation(input, weight, bias, stride, training)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, weight, bias = ctx.saved_tensors

        # Retrieve non-tensor data
        stride = ctx.stride
        training = ctx.training

        # Compute gradients
        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = compute_input_gradient(grad_output, weight, stride)
        if ctx.needs_input_grad[1]:
            grad_weight = compute_weight_gradient(grad_output, input, stride)
        if ctx.needs_input_grad[2]:
            grad_bias = compute_bias_gradient(grad_output)

        # Return gradients (None for stride and training - they're not tensors)
        return grad_input, grad_weight, grad_bias, None, None
```

## Gradient Computation Patterns

Understanding common gradient patterns helps implement backward() correctly.

### Pattern 1: Element-wise Operations

Forward: `y = f(x)` (element-wise function)
Backward: `grad_x = grad_y * f'(x)` (element-wise multiply)

```python
# Example: Custom ReLU
class CustomReLU(Function):
    @staticmethod
    def forward(ctx, input):
        # Save input to compute derivative in backward
        ctx.save_for_backward(input)
        # ReLU: max(0, x)
        output = input.clamp(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Derivative: 1 if x > 0, else 0
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Example: Custom Sigmoid
class CustomSigmoid(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.sigmoid(input)
        # Save output (more efficient than recomputing)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        # Derivative: sigmoid(x) * (1 - sigmoid(x))
        grad_input = grad_output * output * (1 - output)
        return grad_input

# Example: Custom Tanh
class CustomTanh(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.tanh(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        # Derivative: 1 - tanh^2(x)
        grad_input = grad_output * (1 - output * output)
        return grad_input
```

### Pattern 2: Matrix Operations (Chain Rule)

Forward: `Y = X @ W` (matrix multiply)
Backward: Apply chain rule with matrix transpose

```python
class CustomLinear(Function):
    @staticmethod
    def forward(ctx, input, weight):
        # Save both tensors for backward
        ctx.save_for_backward(input, weight)
        # Forward: Y = X @ W
        output = input.mm(weight.t())  # (batch, in) @ (out, in).T = (batch, out)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        # Gradient w.r.t. input: dL/dX = dL/dY @ W
        # Shapes: (batch, out) @ (out, in) = (batch, in) ✓
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        # Gradient w.r.t. weight: dL/dW = dL/dY.T @ X
        # Then transpose to match weight shape
        # Shapes: (out, batch) @ (batch, in) = (out, in) ✓
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        return grad_input, grad_weight

# More complex: Matrix multiply with both inputs requiring gradients
class CustomMatmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        # Forward: C = A @ B
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None

        # Gradient w.r.t. A: dL/dA = dL/dC @ B.T
        if ctx.needs_input_grad[0]:
            grad_a = torch.matmul(grad_output, b.transpose(-2, -1))

        # Gradient w.r.t. B: dL/dB = A.T @ dL/dC
        if ctx.needs_input_grad[1]:
            grad_b = torch.matmul(a.transpose(-2, -1), grad_output)

        return grad_a, grad_b
```

### Pattern 3: Broadcasting Operations

Forward: Operation with broadcasting (e.g., adding bias)
Backward: Sum over broadcasted dimensions

```python
class CustomBiasAdd(Function):
    @staticmethod
    def forward(ctx, input, bias):
        # input: (batch, channels, height, width)
        # bias: (channels,)
        ctx.save_for_backward(input, bias)
        # Broadcasting adds bias to each channel
        output = input + bias.view(1, -1, 1, 1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        grad_input = grad_bias = None

        # Gradient w.r.t. input: just pass through
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        # Gradient w.r.t. bias: sum over broadcasted dimensions
        # grad_output: (batch, channels, height, width)
        # grad_bias should be: (channels,)
        # Sum over batch (0), height (2), width (3)
        if ctx.needs_input_grad[1]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_bias

# General broadcasting pattern
class CustomBroadcastOp(Function):
    @staticmethod
    def forward(ctx, input, param):
        # Save shapes to determine broadcast dimensions
        ctx.input_shape = input.shape
        ctx.param_shape = param.shape
        ctx.save_for_backward(input, param)

        # Some operation with broadcasting
        output = input * param  # param broadcasts to input shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, param = ctx.saved_tensors
        grad_input = grad_param = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output * param

        if ctx.needs_input_grad[1]:
            # Sum grad_output over dimensions that were broadcasted
            grad_param = grad_output * input

            # Find which dimensions were broadcasted
            # Sum over those dimensions to match param shape
            ndim_diff = len(ctx.input_shape) - len(ctx.param_shape)
            for i in range(ndim_diff):
                grad_param = grad_param.sum(0)  # Sum leading dimensions

            for i, (input_dim, param_dim) in enumerate(
                zip(ctx.input_shape[ndim_diff:], ctx.param_shape)
            ):
                if param_dim == 1 and input_dim > 1:
                    grad_param = grad_param.sum(i, keepdim=True)

        return grad_input, grad_param
```

### Pattern 4: Reduction Operations

Forward: Reduce dimensions (sum, mean, max, etc.)
Backward: Expand gradient back to original shape

```python
class CustomSum(Function):
    @staticmethod
    def forward(ctx, input, dim, keepdim=False):
        ctx.input_shape = input.shape
        ctx.dim = dim
        ctx.keepdim = keepdim
        # Sum along dimension
        output = input.sum(dim=dim, keepdim=keepdim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient of sum: distribute grad_output to all elements
        grad_input = grad_output

        # Expand back to original shape
        if not ctx.keepdim:
            # Add back the reduced dimension
            grad_input = grad_input.unsqueeze(ctx.dim)

        # Expand to original shape (broadcasts the gradient)
        grad_input = grad_input.expand(ctx.input_shape)

        return grad_input, None, None

class CustomMean(Function):
    @staticmethod
    def forward(ctx, input, dim, keepdim=False):
        ctx.input_shape = input.shape
        ctx.dim = dim
        ctx.keepdim = keepdim
        output = input.mean(dim=dim, keepdim=keepdim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient of mean: distribute evenly to all elements
        grad_input = grad_output

        if not ctx.keepdim:
            grad_input = grad_input.unsqueeze(ctx.dim)

        grad_input = grad_input.expand(ctx.input_shape)

        # Divide by number of elements that were averaged
        n = ctx.input_shape[ctx.dim]
        grad_input = grad_input / n

        return grad_input, None, None

class CustomMax(Function):
    @staticmethod
    def forward(ctx, input, dim, keepdim=False):
        # Save both max values and indices
        output, indices = input.max(dim=dim, keepdim=keepdim)
        ctx.save_for_backward(indices)
        ctx.input_shape = input.shape
        ctx.dim = dim
        ctx.keepdim = keepdim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors

        # Only the maximum element gets gradient
        grad_input = torch.zeros(ctx.input_shape, device=grad_output.device)

        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(ctx.dim)
            indices = indices.unsqueeze(ctx.dim)

        # Scatter gradient to max indices
        grad_input.scatter_(ctx.dim, indices, grad_output)

        return grad_input, None, None
```

### Pattern 5: Convolution-like Operations

Complex operations that involve multiple dimensions and strides.

```python
class CustomConv1d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0):
        # Use PyTorch's conv for forward
        output = torch.nn.functional.conv1d(
            input, weight, bias, stride, padding
        )

        # Save what's needed for backward
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        grad_input = grad_weight = grad_bias = None

        # Gradient w.r.t. input: convolve grad_output with weight
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv1d_input(
                input.shape, weight, grad_output, stride, padding
            )

        # Gradient w.r.t. weight: convolve input with grad_output
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv1d_weight(
                input, weight.shape, grad_output, stride, padding
            )

        # Gradient w.r.t. bias: sum grad_output over batch and spatial dims
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2))

        return grad_input, grad_weight, grad_bias, None, None
```

## Numerical Gradient Verification (MANDATORY)

**NEVER skip this step.** `gradcheck` verifies your gradient computation is correct by comparing analytical gradients (your backward()) against numerical gradients (finite differences).

### Why gradcheck is Mandatory

```python
# You implement backward() and it "looks right"
class MyFunction(Function):
    @staticmethod
    def backward(ctx, grad_output):
        # Gradient formula looks correct mathematically
        grad = some_computation()
        return grad

# But:
# ❌ Transpose is wrong
# ❌ Shape doesn't match
# ❌ Sign is flipped
# ❌ Missing a term
# ❌ Broadcasting is incorrect

# These bugs are invisible without gradcheck!
# Your model trains but produces wrong results.
# Debugging takes days without knowing gradients are wrong.
```

### Basic gradcheck Usage

```python
import torch
from torch.autograd import gradcheck

def test_my_function():
    """Test custom function with gradcheck."""

    # Create test inputs with requires_grad=True
    # Use double precision for numerical stability
    input = torch.randn(20, 20, dtype=torch.double, requires_grad=True)
    weight = torch.randn(30, 20, dtype=torch.double, requires_grad=True)

    # Run gradcheck
    test = gradcheck(
        MyCustomFunction.apply,  # Your function
        (input, weight),  # Tuple of inputs
        eps=1e-6,  # Finite difference step size
        atol=1e-4,  # Absolute tolerance
        rtol=1e-3,  # Relative tolerance
        raise_exception=True  # Raise error on failure (recommended)
    )

    if test:
        print("✅ Gradient check PASSED!")
    else:
        print("❌ Gradient check FAILED!")
        raise AssertionError("Gradient check failed")

# Run before using your function
test_my_function()
```

### Complete gradcheck Pattern

```python
import torch
from torch.autograd import gradcheck, gradgradcheck

class MyCustomFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

def test_my_custom_function():
    """Comprehensive gradient testing."""

    # Test 1: Basic gradcheck
    print("Test 1: Basic gradient check...")
    input = torch.randn(10, 5, dtype=torch.double, requires_grad=True)
    weight = torch.randn(3, 5, dtype=torch.double, requires_grad=True)
    bias = torch.randn(3, dtype=torch.double, requires_grad=True)

    assert gradcheck(
        MyCustomFunction.apply,
        (input, weight, bias),
        eps=1e-6,
        atol=1e-4,
        raise_exception=True
    ), "Basic gradcheck failed"
    print("✅ Basic gradcheck passed")

    # Test 2: Without bias (optional parameter)
    print("Test 2: Gradient check without bias...")
    assert gradcheck(
        MyCustomFunction.apply,
        (input, weight, None),
        eps=1e-6,
        atol=1e-4,
        raise_exception=True
    ), "Gradcheck without bias failed"
    print("✅ Gradcheck without bias passed")

    # Test 3: Different input shapes
    print("Test 3: Different input shapes...")
    input_large = torch.randn(50, 20, dtype=torch.double, requires_grad=True)
    weight_large = torch.randn(10, 20, dtype=torch.double, requires_grad=True)
    bias_large = torch.randn(10, dtype=torch.double, requires_grad=True)

    assert gradcheck(
        MyCustomFunction.apply,
        (input_large, weight_large, bias_large),
        eps=1e-6,
        atol=1e-4,
        raise_exception=True
    ), "Gradcheck with large inputs failed"
    print("✅ Gradcheck with different shapes passed")

    # Test 4: Second-order gradients (if needed)
    print("Test 4: Second-order gradient check...")
    try:
        assert gradgradcheck(
            MyCustomFunction.apply,
            (input, weight, bias),
            eps=1e-6,
            atol=1e-4,
            raise_exception=True
        ), "Second-order gradcheck failed"
        print("✅ Second-order gradcheck passed")
    except NotImplementedError:
        print("⚠️  Second-order gradients not implemented (OK if not needed)")

    print("\n🎉 All gradient checks passed!")

# ALWAYS run this before using your function in training
test_my_custom_function()
```

### gradcheck Parameters

```python
gradcheck(
    func,              # Your Function.apply
    inputs,            # Tuple of input tensors
    eps=1e-6,         # Finite difference step: f(x+eps) - f(x-eps)
    atol=1e-5,        # Absolute tolerance for comparison
    rtol=1e-3,        # Relative tolerance for comparison
    raise_exception=True,  # Raise on failure (recommended for testing)
    check_sparse_nnz=False,  # Check sparse tensor non-zeros
    nondet_tol=0.0,   # Tolerance for non-deterministic operations
    check_undefined_grad=True,  # Check that undefined grads are None
    check_grad_dtypes=True,  # Check gradient dtypes match
)

# Key insights:
# - Use double precision (dtype=torch.double) for numerical stability
# - eps=1e-6 is good default; smaller for more precision, larger for stability
# - atol/rtol balance: looser tolerances for complex operations
# - raise_exception=True catches bugs immediately in testing
```

### Debugging gradcheck Failures

```python
def debug_gradcheck():
    """Step-by-step debugging when gradcheck fails."""

    # Step 1: Check forward pass works
    print("Step 1: Verify forward pass...")
    input = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    weight = torch.randn(4, 3, dtype=torch.double, requires_grad=True)

    output = MyCustomFunction.apply(input, weight)
    print(f"Output shape: {output.shape}")
    print(f"Output contains NaN: {torch.isnan(output).any()}")
    print(f"Output contains Inf: {torch.isinf(output).any()}")
    assert output.shape == (5, 4), "Forward shape wrong"
    assert not torch.isnan(output).any(), "Forward produces NaN"

    # Step 2: Check backward runs without error
    print("\nStep 2: Verify backward runs...")
    loss = output.sum()
    loss.backward()
    print(f"Input grad shape: {input.grad.shape}")
    print(f"Weight grad shape: {weight.grad.shape}")
    assert input.grad.shape == input.shape, "Input gradient shape mismatch"
    assert weight.grad.shape == weight.shape, "Weight gradient shape mismatch"

    # Step 3: Check gradient magnitudes
    print("\nStep 3: Check gradient magnitudes...")
    print(f"Input grad: mean={input.grad.mean():.6f}, std={input.grad.std():.6f}")
    print(f"Weight grad: mean={weight.grad.mean():.6f}, std={weight.grad.std():.6f}")
    # Should be reasonable numbers (not 1e10 or 1e-20)

    # Step 4: Manual numerical gradient check for one element
    print("\nStep 4: Manual gradient check for one element...")
    input_test = torch.randn(3, 2, dtype=torch.double)
    weight_test = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

    eps = 1e-6

    # Analytical gradient
    output = MyCustomFunction.apply(input_test, weight_test)
    loss = output.sum()
    loss.backward()
    analytical_grad = weight_test.grad[0, 0].item()

    # Numerical gradient (finite difference)
    weight_test_plus = weight_test.clone().detach()
    weight_test_plus[0, 0] += eps
    output_plus = MyCustomFunction.apply(input_test, weight_test_plus)
    loss_plus = output_plus.sum()

    weight_test_minus = weight_test.clone().detach()
    weight_test_minus[0, 0] -= eps
    output_minus = MyCustomFunction.apply(input_test, weight_test_minus)
    loss_minus = output_minus.sum()

    numerical_grad = (loss_plus - loss_minus) / (2 * eps)
    numerical_grad = numerical_grad.item()

    print(f"Analytical gradient: {analytical_grad:.10f}")
    print(f"Numerical gradient:  {numerical_grad:.10f}")
    print(f"Difference:          {abs(analytical_grad - numerical_grad):.10e}")

    if abs(analytical_grad - numerical_grad) > 1e-4:
        print("❌ Large difference - gradient implementation likely wrong")
    else:
        print("✅ Small difference - gradient likely correct")

    # Step 5: Run gradcheck with verbose output
    print("\nStep 5: Run gradcheck...")
    input_check = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
    weight_check = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

    try:
        result = gradcheck(
            MyCustomFunction.apply,
            (input_check, weight_check),
            eps=1e-6,
            atol=1e-4,
            raise_exception=True
        )
        print("✅ gradcheck passed!")
    except RuntimeError as e:
        print(f"❌ gradcheck failed with error:\n{e}")
        # Error message shows which gradient failed and by how much

debug_gradcheck()
```

### Common gradcheck Failure Reasons

```python
# Failure 1: Wrong gradient formula
class WrongGradient(Function):
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        # ❌ WRONG: No transpose
        grad_input = grad_output @ weight  # Should be weight.t()
        return grad_input, None
# gradcheck fails: analytical ≠ numerical

# Failure 2: Shape mismatch
class WrongShape(Function):
    @staticmethod
    def backward(ctx, grad_output):
        # ❌ WRONG: Returns wrong shape
        return grad_output.sum(), None  # Should be grad_output.shape == input.shape
# gradcheck fails: shape error

# Failure 3: In-place operation
class InplaceOperation(Function):
    @staticmethod
    def backward(ctx, grad_output):
        grad_output[grad_output < 0] = 0  # ❌ IN-PLACE
        return grad_output
# gradcheck fails: modified by inplace operation

# Failure 4: Not using saved tensors correctly
class WrongSaved(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input.clone())  # Saved clone
        return input * 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Using saved tensor is OK, but if logic depends on
        # input's original properties and clone loses them, fails
        return grad_output * 2
# May pass or fail depending on what was lost in clone

# Failure 5: Forgot to return gradients for all inputs
class MissingGradient(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        # ...
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # ❌ WRONG: Only returns 2 gradients for 3 inputs
        return grad_input, grad_weight
# gradcheck fails: tuple length mismatch
```

## Common Pitfalls

### Pitfall 1: In-Place Operations

In-place operations modify tensors that other operations depend on, breaking autograd.

```python
# ❌ WRONG: In-place operation in forward
class InplaceForward(Function):
    @staticmethod
    def forward(ctx, input):
        input[input < 0] = 0  # IN-PLACE - breaks autograd!
        return input

# ❌ WRONG: In-place operation in backward
class InplaceBackward(Function):
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output[input < 0] = 0  # IN-PLACE - breaks autograd!
        return grad_output

# ✅ CORRECT: Create new tensor
class CorrectInplace(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()  # Create new tensor
        output[output < 0] = 0  # Modify copy, not original
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()  # Create new tensor
        grad_input[input < 0] = 0  # Modify copy
        return grad_input

# Even better: Use non-in-place operations
class BestInplace(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)  # Non-in-place

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (input > 0).float()  # Non-in-place
        return grad_input
```

### Pitfall 2: Gradient Shape Mismatch

Gradient must match input shape exactly.

```python
# ❌ WRONG: Gradient shape doesn't match input
class WrongShape(Function):
    @staticmethod
    def forward(ctx, input):
        # input: (32, 128)
        ctx.save_for_backward(input)
        return input.sum()  # scalar

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # grad_output is scalar, but need (32, 128)
        return grad_output  # ❌ WRONG: scalar ≠ (32, 128)

# ✅ CORRECT: Expand to match input shape
class CorrectShape(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input_shape = input.shape
        return input.sum()

    @staticmethod
    def backward(ctx, grad_output):
        # Expand scalar to input shape
        grad_input = grad_output.expand(ctx.input_shape)
        return grad_input

# Always verify shapes
def verify_shapes(ctx, grad_output, *grad_inputs):
    """Helper to verify gradient shapes."""
    for i, (grad, tensor) in enumerate(zip(grad_inputs, ctx.saved_tensors)):
        if grad is not None:
            assert grad.shape == tensor.shape, \
                f"Gradient {i} shape {grad.shape} != input shape {tensor.shape}"
```

### Pitfall 3: Not Checking needs_input_grad

Computing gradients when not needed wastes computation.

```python
# ❌ WASTEFUL: Always compute all gradients
class AlwaysCompute(Function):
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        # Compute all gradients even if not needed
        grad_input = expensive_computation_1(grad_output, weight)
        grad_weight = expensive_computation_2(grad_output, input)
        grad_bias = expensive_computation_3(grad_output)

        return grad_input, grad_weight, grad_bias

# ✅ EFFICIENT: Check needs_input_grad first
class EfficientCompute(Function):
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Only compute if needed
        if ctx.needs_input_grad[0]:
            grad_input = expensive_computation_1(grad_output, weight)

        if ctx.needs_input_grad[1]:
            grad_weight = expensive_computation_2(grad_output, input)

        if ctx.needs_input_grad[2]:
            grad_bias = expensive_computation_3(grad_output)

        return grad_input, grad_weight, grad_bias

# Example where it matters
def example_needs_input_grad():
    """Demonstrate needs_input_grad optimization."""

    input = torch.randn(100, 100, requires_grad=True)
    weight = torch.randn(100, 100, requires_grad=False)  # No gradient needed

    # Without check: computes grad_weight unnecessarily
    # With check: skips grad_weight computation (faster)

    output = MyFunction.apply(input, weight)
    loss = output.sum()
    loss.backward()

    # weight.grad is None because requires_grad=False
    assert weight.grad is None
```

### Pitfall 4: Using .data Instead of .detach()

`.data` bypasses autograd tracking incorrectly.

```python
# ❌ WRONG: Using .data
class UsingData(Function):
    @staticmethod
    def forward(ctx, input):
        # .data returns tensor without autograd tracking
        # But doesn't properly detach from computation graph
        ctx.save_for_backward(input.data)  # ❌ WRONG
        return input * 2

    @staticmethod
    def backward(ctx, grad_output):
        input_data, = ctx.saved_tensors
        # May produce incorrect gradients
        return grad_output * 2

# ✅ CORRECT: Use .detach() or save normally
class UsingDetach(Function):
    @staticmethod
    def forward(ctx, input):
        # If you need to save without tracking gradient
        ctx.save_for_backward(input.detach())  # ✅ Properly detaches
        # Or just save normally if gradient tracking is OK
        ctx.save_for_backward(input)  # ✅ Most common
        return input * 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2

# When to detach in custom functions
class WhenToDetach(Function):
    @staticmethod
    def forward(ctx, input, target):
        # Save input normally (gradient needed)
        # Detach target (gradient not needed, just reference)
        ctx.save_for_backward(input, target.detach())

        loss = (input - target).pow(2).mean()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        # Compute gradient w.r.t. input only
        grad_input = 2 * (input - target) * grad_output
        return grad_input, None
```

### Pitfall 5: Modifying grad_output

Never modify grad_output in-place; it's used by other operations.

```python
# ❌ WRONG: Modifying grad_output in-place
class ModifyGradOutput(Function):
    @staticmethod
    def backward(ctx, grad_output):
        # ❌ WRONG: In-place modification
        grad_output *= 2
        return grad_output

# ✅ CORRECT: Create new tensor
class DontModifyGradOutput(Function):
    @staticmethod
    def backward(ctx, grad_output):
        # ✅ Create new tensor
        grad_input = grad_output * 2
        return grad_input

# Why it matters
def why_grad_output_matters():
    """Demonstrate why modifying grad_output breaks autograd."""

    # Consider: z = f(g(x))
    # Backward: dz/dx = dz/dg * dg/dx
    #
    # If f.backward() modifies grad_output (dz/dg),
    # then g.backward() receives wrong gradient!

    x = torch.randn(5, requires_grad=True)

    # g(x)
    y = x * 2

    # f(g(x)) - uses custom function that modifies grad_output
    z = BadFunction.apply(y)

    z.backward()

    # x.grad is now WRONG because BadFunction modified grad_output
    # that was passed to y's backward
```

### Pitfall 6: Forgetting to Return None for Non-Tensor Arguments

Must return gradient for every forward() argument.

```python
# ❌ WRONG: Not enough return values
class NotEnoughReturns(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride):
        # 3 arguments (excluding ctx)
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        return some_operation(input, kernel_size, stride)

    @staticmethod
    def backward(ctx, grad_output):
        # ❌ WRONG: Only returns 1 value for 3 inputs
        return grad_output  # Crashes: expected 3 values

# ✅ CORRECT: Return gradient (or None) for each input
class EnoughReturns(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride):
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        return some_operation(input, kernel_size, stride)

    @staticmethod
    def backward(ctx, grad_output):
        # ✅ Return 3 values: grad for input, None for kernel_size, None for stride
        return grad_output, None, None

# Rule of thumb
def count_returns():
    """
    backward() must return one value per forward() argument (excluding ctx).

    forward(ctx, a, b, c, d=None) → backward must return (grad_a, grad_b, grad_c, grad_d)

    Use None for:
    - Non-tensor arguments (ints, strings, etc.)
    - Optional arguments that were None
    - Tensors that don't need gradients
    """
    pass
```

### Pitfall 7: Incorrect Broadcasting in Gradient

Gradient must account for broadcasting that occurred in forward.

```python
# ❌ WRONG: Doesn't handle broadcasting correctly
class WrongBroadcast(Function):
    @staticmethod
    def forward(ctx, input, weight):
        # input: (32, 64, 10, 10)
        # weight: (64, 1, 1) - broadcasts to (32, 64, 10, 10)
        ctx.save_for_backward(input, weight)
        output = input * weight  # Broadcasting happens
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_input = grad_output * weight  # ✅ This is fine

        # ❌ WRONG: grad_weight shape is (32, 64, 10, 10), should be (64, 1, 1)
        grad_weight = grad_output * input
        return grad_input, grad_weight  # Shape mismatch!

# ✅ CORRECT: Sum over broadcasted dimensions
class CorrectBroadcast(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        ctx.input_shape = input.shape
        ctx.weight_shape = weight.shape
        output = input * weight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_input = grad_output * weight

        # ✅ Sum over dimensions that were broadcasted
        grad_weight = grad_output * input
        # weight shape: (64, 1, 1), grad_weight current: (32, 64, 10, 10)
        # Sum over batch (0), height (2), width (3)
        grad_weight = grad_weight.sum(dim=(0, 2, 3), keepdim=True)
        # Now grad_weight shape: (1, 64, 1, 1)
        # Squeeze batch dimension: (64, 1, 1) ✅
        grad_weight = grad_weight.squeeze(0)

        return grad_input, grad_weight

# General broadcasting gradient pattern
def sum_to_shape(tensor, shape):
    """Sum tensor to match target shape (handles broadcasting)."""
    # Find dimensions that were added
    while tensor.dim() > len(shape):
        tensor = tensor.sum(0)

    # Find dimensions that were size 1 and got broadcasted
    for i, (t_dim, s_dim) in enumerate(zip(tensor.shape, shape)):
        if s_dim == 1 and t_dim > 1:
            tensor = tensor.sum(i, keepdim=True)

    return tensor

class GeneralBroadcast(Function):
    @staticmethod
    def backward(ctx, grad_output):
        input, param = ctx.saved_tensors

        grad_input = grad_output * param

        grad_param = grad_output * input
        # Sum to match param's original shape
        grad_param = sum_to_shape(grad_param, param.shape)

        return grad_input, grad_param
```

## Memory Efficiency Patterns

Custom functions enable memory optimizations not possible with standard autograd.

### Pattern 1: Gradient Checkpointing

Trade computation for memory by recomputing forward in backward.

```python
class CheckpointedFunction(Function):
    """
    Gradient checkpointing: Don't save activations, recompute in backward.

    Memory: O(1) instead of O(n) for n layers
    Time: 2x forward pass (once in forward, once in backward)
    """

    @staticmethod
    def forward(ctx, input, *args):
        # Save inputs and parameters, NOT intermediate activations
        ctx.save_for_backward(input, *args)

        # Compute forward pass (activations not saved)
        # For complex operations, this may compute many intermediate values
        output = expensive_computation(input, *args)

        # Intermediate activations are garbage collected
        # Saves memory!
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve inputs
        input, *args = ctx.saved_tensors

        # Recompute forward pass to get intermediate values
        # This time, track gradients
        with torch.enable_grad():
            # Detach input, then set requires_grad
            # (Required for computing gradients)
            input = input.detach().requires_grad_(True)

            # Recompute forward
            output = expensive_computation(input, *args)

        # Now compute gradients using autograd
        grad_input, = torch.autograd.grad(
            outputs=output,
            inputs=input,
            grad_outputs=grad_output,
            retain_graph=False
        )

        # Return gradient for input (and None for args if they're parameters)
        return (grad_input,) + (None,) * len(args)

# Example: Checkpointed Sequential Layers
class CheckpointedSequential(Function):
    @staticmethod
    def forward(ctx, input, *layers):
        """
        Forward through multiple layers without saving intermediate activations.

        Normal: Saves n-1 activations for n layers
        Checkpointed: Saves only input and parameters
        """
        ctx.layers = layers
        ctx.save_for_backward(input)

        # Forward through all layers
        output = input
        for layer in layers:
            output = layer(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        layers = ctx.layers

        # Recompute forward to get intermediate activations
        with torch.enable_grad():
            input = input.detach().requires_grad_(True)

            # Forward pass again, this time tracking activations
            activations = [input]
            output = input
            for layer in layers:
                output = layer(output)
                activations.append(output)

        # Backward through layers
        grad = grad_output
        for i in reversed(range(len(layers))):
            # Compute gradient through layer i
            grad, = torch.autograd.grad(
                outputs=activations[i+1],
                inputs=activations[i],
                grad_outputs=grad,
                retain_graph=True
            )

        grad_input = grad

        # No gradients for layers (they're not inputs to forward)
        return (grad_input,) + (None,) * len(layers)

# PyTorch provides torch.utils.checkpoint.checkpoint for this
# But understanding the pattern helps for custom cases
```

### Pattern 2: Selective Saving

Only save what's needed for backward; recompute or omit the rest.

```python
class SelectiveSaving(Function):
    """
    Save only essential tensors; recompute others in backward.
    """

    @staticmethod
    def forward(ctx, input, weight, bias):
        # Compute intermediate values
        weighted = input @ weight
        activated = torch.relu(weighted)
        output = activated + bias

        # DON'T save everything:
        # ctx.save_for_backward(input, weight, bias, weighted, activated)
        # ❌ Saves 5 tensors

        # ✅ SAVE ONLY WHAT'S NEEDED:
        # Can recompute 'weighted' from input and weight
        # Can recompute 'activated' from weighted
        ctx.save_for_backward(input, weight, bias, activated)
        # ✅ Saves 4 tensors (or even fewer)

        # Or even more selective:
        # ctx.save_for_backward(input, weight, bias)
        # ctx.save_for_backward(activated > 0)  # Save mask, not full tensor

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, activated = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        # Gradient through bias addition
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        # Gradient through ReLU (need activation mask)
        grad_activated = grad_output.clone()
        grad_activated[activated <= 0] = 0

        # Gradient through matmul
        if ctx.needs_input_grad[0]:
            grad_input = grad_activated @ weight.t()
        if ctx.needs_input_grad[1]:
            grad_weight = input.t() @ grad_activated

        return grad_input, grad_weight, grad_bias

# Even more selective: Save boolean mask instead of full tensor
class UltraSelective(Function):
    @staticmethod
    def forward(ctx, input, weight):
        output = torch.relu(input @ weight)

        # Instead of saving full 'output' tensor:
        # ctx.save_for_backward(input, weight, output)  # Large memory

        # ✅ Save only boolean mask (1 bit per element vs 32 bits)
        ctx.save_for_backward(input, weight)
        ctx.relu_mask = (output > 0)  # Boolean tensor (much smaller)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        relu_mask = ctx.relu_mask

        # Use mask to apply ReLU gradient
        grad_weighted = grad_output * relu_mask.float()

        grad_input = grad_weighted @ weight.t() if ctx.needs_input_grad[0] else None
        grad_weight = input.t() @ grad_weighted if ctx.needs_input_grad[1] else None

        return grad_input, grad_weight
```

### Pattern 3: Detaching Tensors That Don't Need Gradients

Detach tensors that are used in forward but don't need gradients.

```python
class DetachPattern(Function):
    """
    Detach tensors that don't need gradient computation.
    """

    @staticmethod
    def forward(ctx, input, target, weight):
        """
        Compute loss between input and target.
        Target doesn't need gradients (it's labels).
        """
        # Save input and weight (need gradients)
        ctx.save_for_backward(input, weight)

        # Detach target (doesn't need gradients)
        # This breaks the autograd connection, saving memory
        ctx.target = target.detach()

        # Compute weighted loss
        loss = ((input - target) ** 2 * weight).mean()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        target = ctx.target  # Already detached

        # Compute gradients
        diff = input - target

        grad_input = None
        grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = 2 * diff * weight * grad_output

        # No gradient for target (ctx.needs_input_grad[1] is False)

        if ctx.needs_input_grad[2]:
            grad_weight = (diff ** 2) * grad_output

        return grad_input, None, grad_weight

# When to detach
def detach_guidelines():
    """
    Detach tensors when:
    1. They're labels/targets (no gradient needed)
    2. They're constants (no gradient needed)
    3. They're from non-differentiable sources
    4. You explicitly don't want gradients to flow through them

    Don't detach when:
    1. Gradient is needed for that tensor
    2. Gradient will flow through that path
    """
    pass
```

## Advanced Patterns

### Pattern 1: Double Backward (Second-Order Derivatives)

Support computing gradients of gradients.

```python
class DoubleBackwardFunction(Function):
    """
    Function that supports double backward for second-order derivatives.

    Example: Hessian computation, meta-learning, some regularization terms.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Forward: y = x^2
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # First derivative: dy/dx = 2x
        grad_input = 2 * input * grad_output

        # For double backward, need to return tensor that supports backward
        # Not just detached scalar
        return grad_input

    # For explicit double backward support (optional, autograd often handles it)
    @staticmethod
    def jvp(ctx, *grad_inputs):
        """
        Jacobian-vector product for forward-mode AD.
        Needed for some second-order derivative computations.
        """
        # Usually not needed; autograd handles it
        pass

# Test double backward
def test_double_backward():
    """Test that double backward works."""

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # First forward
    y = DoubleBackwardFunction.apply(x)

    # First backward: dy/dx
    grad_x, = torch.autograd.grad(y.sum(), x, create_graph=True)
    # grad_x = 2x = [2, 4, 6]

    # Second backward: d(dy/dx)/dx = d(2x)/dx = 2
    grad_grad_x, = torch.autograd.grad(grad_x.sum(), x)
    # grad_grad_x = [2, 2, 2]

    print(f"First derivative: {grad_x}")   # [2, 4, 6]
    print(f"Second derivative: {grad_grad_x}")  # [2, 2, 2]

    assert torch.allclose(grad_x, 2 * x)
    assert torch.allclose(grad_grad_x, torch.ones_like(x) * 2)
    print("✅ Double backward works!")

test_double_backward()

# Example: Function where double backward matters
class HessianVectorProduct(Function):
    """
    Efficiently compute Hessian-vector product: H @ v
    where H = ∇²f(x) is the Hessian matrix.

    Used in: second-order optimization, meta-learning.
    """

    @staticmethod
    def forward(ctx, input, vector):
        ctx.save_for_backward(input, vector)
        # Placeholder forward (actual computation in backward)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, vector = ctx.saved_tensors

        # Compute gradient
        grad_input = grad_output

        # This gradient can be backpropagated through again
        # for Hessian computation
        return grad_input, None

# Using double backward for Hessian
def compute_hessian(f, x):
    """
    Compute Hessian of scalar function f at point x.

    Uses double backward: H[i,j] = ∂²f/∂x_i∂x_j
    """
    # First derivative
    grad_x, = torch.autograd.grad(f(x), x, create_graph=True)

    # Second derivative (Hessian)
    hessian = []
    for i in range(x.shape[0]):
        grad_grad_x, = torch.autograd.grad(
            grad_x[i], x, retain_graph=True
        )
        hessian.append(grad_grad_x)

    return torch.stack(hessian)
```

### Pattern 2: Custom Backward Hooks

Modify gradients during backward pass without changing the function.

```python
def gradient_clipping_hook(grad):
    """
    Hook to clip gradients to [-1, 1].
    Applied to tensor, not Function.
    """
    return torch.clamp(grad, -1, 1)

def gradient_noise_hook(grad):
    """Add noise to gradients (for regularization)."""
    noise = torch.randn_like(grad) * 0.01
    return grad + noise

def gradient_logging_hook(grad):
    """Log gradient statistics."""
    print(f"Gradient: mean={grad.mean():.6f}, std={grad.std():.6f}, max={grad.abs().max():.6f}")
    return grad  # Return unchanged

# Using hooks
def use_hooks():
    """Example of using gradient hooks."""

    input = torch.randn(10, 10, requires_grad=True)
    weight = torch.randn(10, 10, requires_grad=True)

    # Register hooks
    input.register_hook(gradient_clipping_hook)
    weight.register_hook(gradient_logging_hook)

    # Forward and backward
    output = MyFunction.apply(input, weight)
    loss = output.sum()
    loss.backward()

    # Hooks are applied during backward
    # input.grad is clipped to [-1, 1]
    # weight.grad statistics are logged

# Hooks in custom functions
class FunctionWithHook(Function):
    @staticmethod
    def forward(ctx, input, clip_grad=False):
        ctx.clip_grad = clip_grad
        ctx.save_for_backward(input)
        return input * 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input = grad_output * 2

        # Apply custom modification based on context
        if ctx.clip_grad:
            grad_input = torch.clamp(grad_input, -1, 1)

        return grad_input, None

# Removing hooks
def manage_hooks():
    """Add and remove hooks."""

    tensor = torch.randn(5, requires_grad=True)

    # Add hook (returns handle)
    hook_handle = tensor.register_hook(gradient_clipping_hook)

    # Use tensor
    loss = (tensor ** 2).sum()
    loss.backward()
    # Hook is applied

    # Remove hook
    hook_handle.remove()

    # Hook no longer applied in subsequent backwards
    tensor.grad.zero_()
    loss = (tensor ** 2).sum()
    loss.backward()
    # Hook NOT applied
```

### Pattern 3: Custom Gradient for Part of Computation

Stop gradients or customize them for specific operations.

```python
class StopGradient(Function):
    """
    Stop gradient flow (like tf.stop_gradient or tensor.detach()).
    Forward: pass through
    Backward: return None (no gradient)
    """

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # Don't pass gradient through
        return None

# Usage: Stop gradient flow
def stop_gradient_example():
    x = torch.randn(5, requires_grad=True)

    # y doesn't get gradients from z
    y = StopGradient.apply(x)
    z = y ** 2
    z.sum().backward()

    assert x.grad is None  # No gradient flowed to x

class StraightThroughEstimator(Function):
    """
    Straight-through estimator for non-differentiable operations.

    Forward: non-differentiable operation (e.g., binarization)
    Backward: pretend it was identity (pass gradient through)

    Used for: quantization, binarization, discrete operations.
    """

    @staticmethod
    def forward(ctx, input):
        # Non-differentiable: binarize to {-1, 1}
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # Pretend forward was identity: gradient passes through unchanged
        return grad_output

# Usage: Train networks with binary weights
def straight_through_example():
    """
    Train a network with binary weights using straight-through estimator.
    """
    weight = torch.randn(10, 10, requires_grad=True)
    input = torch.randn(32, 10)

    # Binarize weight for forward pass
    binary_weight = StraightThroughEstimator.apply(weight)
    # binary_weight ∈ {-1, 1}

    # Use binary weight
    output = input @ binary_weight
    loss = output.sum()
    loss.backward()

    # weight.grad exists (even though sign() isn't differentiable)
    # Gradient passed through as if sign() was identity
    assert weight.grad is not None

class CustomGradientScale(Function):
    """
    Scale gradient by a factor without changing forward.

    Forward: pass through
    Backward: scale gradient by alpha

    Used for: gradient reversal layers (adversarial training),
             controlling gradient flow in different branches.
    """

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # Scale gradient
        return grad_output * ctx.alpha, None

# Usage: Gradient reversal layer
def gradient_reversal_layer(input, alpha=-1.0):
    """
    Reverses gradients (multiplies by -1).
    Used in domain adaptation for adversarial training.
    """
    return CustomGradientScale.apply(input, alpha)
```

## Complete Real-World Examples

### Example 1: Custom Swish Activation with Learnable Beta

```python
import torch
from torch.autograd import Function, gradcheck
import torch.nn as nn

class SwishFunction(Function):
    """
    Custom Swish activation: f(x) = x * sigmoid(β * x)
    where β is a learnable parameter.

    Forward: y = x * σ(βx)
    Backward: dy/dx = σ(βx) + x * σ(βx) * (1 - σ(βx)) * β
              dy/dβ = x² * σ(βx) * (1 - σ(βx))
    """

    @staticmethod
    def forward(ctx, input, beta):
        """
        Args:
            input: Input tensor
            beta: Learnable scaling parameter (scalar tensor)
        Returns:
            output: Swish activation output
        """
        ctx.save_for_backward(input, beta)

        # Compute sigmoid(beta * input)
        sigmoid_beta_input = torch.sigmoid(beta * input)

        # Save for backward (more efficient than recomputing)
        ctx.save_for_backward(input, beta, sigmoid_beta_input)

        # f(x) = x * sigmoid(beta * x)
        output = input * sigmoid_beta_input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradients using chain rule.
        """
        input, beta, sigmoid_beta_input = ctx.saved_tensors

        grad_input = grad_beta = None

        # Gradient w.r.t. input
        if ctx.needs_input_grad[0]:
            # d/dx[x * σ(βx)] = σ(βx) + x * σ'(βx) * β
            # where σ'(z) = σ(z) * (1 - σ(z))
            sigmoid_derivative = sigmoid_beta_input * (1 - sigmoid_beta_input)
            grad_input = grad_output * (
                sigmoid_beta_input + input * sigmoid_derivative * beta
            )

        # Gradient w.r.t. beta
        if ctx.needs_input_grad[1]:
            # d/dβ[x * σ(βx)] = x * σ'(βx) * x = x² * σ(βx) * (1 - σ(βx))
            sigmoid_derivative = sigmoid_beta_input * (1 - sigmoid_beta_input)
            grad_beta = grad_output * (input ** 2) * sigmoid_derivative
            # Sum over all elements (beta is scalar)
            grad_beta = grad_beta.sum()

        return grad_input, grad_beta

class Swish(nn.Module):
    """
    Swish activation module with learnable beta parameter.
    """

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, input):
        return SwishFunction.apply(input, self.beta)

# Test with gradcheck
def test_swish():
    """Verify Swish implementation with gradcheck."""
    print("Testing Swish activation...")

    # Test 1: Basic gradcheck
    input = torch.randn(10, 5, dtype=torch.double, requires_grad=True)
    beta = torch.tensor(1.0, dtype=torch.double, requires_grad=True)

    assert gradcheck(
        SwishFunction.apply,
        (input, beta),
        eps=1e-6,
        atol=1e-4,
        raise_exception=True
    ), "Swish gradcheck failed"
    print("✅ Basic gradcheck passed")

    # Test 2: Different beta values
    for beta_val in [0.5, 1.0, 2.0]:
        beta = torch.tensor(beta_val, dtype=torch.double, requires_grad=True)
        assert gradcheck(
            SwishFunction.apply,
            (input, beta),
            eps=1e-6,
            atol=1e-4,
            raise_exception=True
        ), f"Swish gradcheck failed for beta={beta_val}"
    print("✅ Multiple beta values passed")

    # Test 3: Use in module
    module = Swish(beta=1.0)
    input_single = torch.randn(32, 128, requires_grad=True)
    output = module(input_single)
    loss = output.sum()
    loss.backward()

    assert input_single.grad is not None
    assert module.beta.grad is not None
    print("✅ Module usage works")

    print("\n🎉 Swish activation fully tested!")

test_swish()

# Usage example
def use_swish_in_model():
    """Use Swish in a neural network."""

    model = nn.Sequential(
        nn.Linear(784, 256),
        Swish(beta=1.0),  # Learnable beta
        nn.Linear(256, 128),
        Swish(beta=1.0),
        nn.Linear(128, 10)
    )

    # Train model...
    # Beta parameters will be learned along with weights
```

### Example 2: Numerically Stable LogSumExp

```python
class LogSumExp(Function):
    """
    Numerically stable log-sum-exp operation.

    Forward: log(sum(exp(x_i)))
    Uses max trick: log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))

    Backward: softmax(x_i)
    """

    @staticmethod
    def forward(ctx, input, dim):
        """
        Args:
            input: Input tensor
            dim: Dimension to sum over
        Returns:
            logsumexp: log(sum(exp(input))) along dim
        """
        # Max trick for numerical stability
        max_val, _ = input.max(dim=dim, keepdim=True)
        input_shifted = input - max_val

        # Compute log-sum-exp
        sumexp = torch.exp(input_shifted).sum(dim=dim, keepdim=True)
        logsumexp = torch.log(sumexp) + max_val

        # Save softmax for backward
        softmax = torch.exp(input_shifted) / sumexp
        ctx.save_for_backward(softmax)
        ctx.dim = dim

        return logsumexp.squeeze(dim)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Gradient of log-sum-exp is softmax.
        """
        softmax, = ctx.saved_tensors

        # Expand grad_output to match softmax shape
        grad_output_expanded = grad_output.unsqueeze(ctx.dim)

        # Gradient: softmax * grad_output
        grad_input = softmax * grad_output_expanded

        return grad_input, None

# Test
def test_logsumexp():
    """Test LogSumExp implementation."""
    print("Testing LogSumExp...")

    input = torch.randn(10, 5, dtype=torch.double, requires_grad=True)
    dim = 1

    # gradcheck
    assert gradcheck(
        lambda x: LogSumExp.apply(x, dim),
        (input,),
        eps=1e-6,
        atol=1e-4,
        raise_exception=True
    ), "LogSumExp gradcheck failed"
    print("✅ LogSumExp gradcheck passed")

    # Compare with PyTorch's implementation
    custom_result = LogSumExp.apply(input, dim)
    torch_result = torch.logsumexp(input, dim=dim)

    assert torch.allclose(custom_result, torch_result, atol=1e-5), \
        "LogSumExp doesn't match PyTorch"
    print("✅ LogSumExp matches PyTorch implementation")

    print("\n🎉 LogSumExp fully tested!")

test_logsumexp()
```

### Example 3: Fused Linear + ReLU (Memory Efficient)

```python
class FusedLinearReLU(Function):
    """
    Fused linear + ReLU operation.
    Saves memory by not materializing intermediate activations.

    Forward: ReLU(X @ W + b)
    Memory: Only saves mask (boolean) and input/weights, not intermediate
    """

    @staticmethod
    def forward(ctx, input, weight, bias):
        """
        Args:
            input: (batch, in_features)
            weight: (out_features, in_features)
            bias: (out_features,)
        Returns:
            output: (batch, out_features)
        """
        # Compute linear
        linear_output = input.mm(weight.t())
        if bias is not None:
            linear_output += bias

        # Apply ReLU and save mask (not full tensor!)
        output = torch.relu(linear_output)
        relu_mask = (linear_output > 0)  # Boolean (1 bit per element)

        # Save only input, weight, bias, and mask
        # NOT saving linear_output (saves memory)
        ctx.save_for_backward(input, weight, bias)
        ctx.relu_mask = relu_mask

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward through ReLU and linear.
        """
        input, weight, bias = ctx.saved_tensors
        relu_mask = ctx.relu_mask

        grad_input = grad_weight = grad_bias = None

        # Gradient through ReLU (use mask)
        grad_linear = grad_output * relu_mask.float()

        # Gradient through linear
        if ctx.needs_input_grad[0]:
            grad_input = grad_linear.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_linear.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_linear.sum(0)

        return grad_input, grad_weight, grad_bias

# Test
def test_fused_linear_relu():
    """Test fused operation."""
    print("Testing FusedLinearReLU...")

    input = torch.randn(20, 10, dtype=torch.double, requires_grad=True)
    weight = torch.randn(15, 10, dtype=torch.double, requires_grad=True)
    bias = torch.randn(15, dtype=torch.double, requires_grad=True)

    # gradcheck
    assert gradcheck(
        FusedLinearReLU.apply,
        (input, weight, bias),
        eps=1e-6,
        atol=1e-4,
        raise_exception=True
    ), "FusedLinearReLU gradcheck failed"
    print("✅ FusedLinearReLU gradcheck passed")

    # Compare with separate operations
    fused_output = FusedLinearReLU.apply(input, weight, bias)

    separate_output = torch.relu(input.mm(weight.t()) + bias)

    assert torch.allclose(fused_output, separate_output, atol=1e-5), \
        "Fused output doesn't match separate operations"
    print("✅ Fused matches separate operations")

    print("\n🎉 FusedLinearReLU fully tested!")

test_fused_linear_relu()
```

## Debugging Custom Functions

### Systematic Debugging Workflow

When your custom function has bugs, follow this workflow:

```python
def debug_custom_function():
    """
    Step-by-step debugging of custom autograd functions.
    """

    print("=== DEBUGGING CUSTOM AUTOGRAD FUNCTION ===\n")

    # Step 1: Verify forward pass works
    print("Step 1: Testing forward pass...")
    try:
        input = torch.randn(5, 3, dtype=torch.double)
        weight = torch.randn(4, 3, dtype=torch.double)

        output = MyCustomFunction.apply(input, weight)

        print(f"✅ Forward pass works")
        print(f"   Input shape: {input.shape}")
        print(f"   Weight shape: {weight.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output contains NaN: {torch.isnan(output).any()}")
        print(f"   Output contains Inf: {torch.isinf(output).any()}")

        # Check expected shape
        expected_shape = (5, 4)  # Based on your operation
        assert output.shape == expected_shape, \
            f"Wrong output shape: {output.shape} != {expected_shape}"
        print(f"   Output shape correct: {expected_shape}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    print()

    # Step 2: Verify backward runs without error
    print("Step 2: Testing backward pass...")
    try:
        input = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
        weight = torch.randn(4, 3, dtype=torch.double, requires_grad=True)

        output = MyCustomFunction.apply(input, weight)
        loss = output.sum()
        loss.backward()

        print(f"✅ Backward pass runs")
        print(f"   Input grad shape: {input.grad.shape}")
        print(f"   Weight grad shape: {weight.grad.shape}")

        # Check gradient shapes
        assert input.grad.shape == input.shape, \
            f"Input gradient shape mismatch: {input.grad.shape} != {input.shape}"
        assert weight.grad.shape == weight.shape, \
            f"Weight gradient shape mismatch: {weight.grad.shape} != {weight.shape}"
        print(f"   Gradient shapes correct")

        # Check for NaN/Inf
        assert not torch.isnan(input.grad).any(), "Input gradient contains NaN"
        assert not torch.isnan(weight.grad).any(), "Weight gradient contains NaN"
        assert not torch.isinf(input.grad).any(), "Input gradient contains Inf"
        assert not torch.isinf(weight.grad).any(), "Weight gradient contains Inf"
        print(f"   Gradients are finite")

    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return

    print()

    # Step 3: Check gradient magnitudes
    print("Step 3: Checking gradient magnitudes...")
    print(f"   Input grad - mean: {input.grad.mean():.6f}, std: {input.grad.std():.6f}, max: {input.grad.abs().max():.6f}")
    print(f"   Weight grad - mean: {weight.grad.mean():.6f}, std: {weight.grad.std():.6f}, max: {weight.grad.abs().max():.6f}")

    # Reasonable gradient magnitudes (problem-dependent)
    if input.grad.abs().max() > 1e6:
        print(f"   ⚠️  Input gradient very large (may indicate bug)")
    if input.grad.abs().max() < 1e-6:
        print(f"   ⚠️  Input gradient very small (may indicate vanishing gradient)")

    print()

    # Step 4: Manual numerical gradient check for one element
    print("Step 4: Manual numerical gradient check...")

    input_test = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
    weight_test = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

    # Analytical gradient
    output = MyCustomFunction.apply(input_test, weight_test)
    loss = output.sum()
    loss.backward()
    analytical_grad = weight_test.grad[0, 0].item()

    # Numerical gradient
    eps = 1e-6

    weight_plus = weight_test.clone().detach()
    weight_plus[0, 0] += eps
    output_plus = MyCustomFunction.apply(input_test, weight_plus)
    loss_plus = output_plus.sum()

    weight_minus = weight_test.clone().detach()
    weight_minus[0, 0] -= eps
    output_minus = MyCustomFunction.apply(input_test, weight_minus)
    loss_minus = output_minus.sum()

    numerical_grad = ((loss_plus - loss_minus) / (2 * eps)).item()

    diff = abs(analytical_grad - numerical_grad)
    print(f"   Analytical gradient: {analytical_grad:.10f}")
    print(f"   Numerical gradient:  {numerical_grad:.10f}")
    print(f"   Absolute difference: {diff:.10e}")

    if diff < 1e-4:
        print(f"   ✅ Small difference - gradient likely correct")
    else:
        print(f"   ❌ Large difference - gradient likely WRONG")

    print()

    # Step 5: Full gradcheck
    print("Step 5: Running full gradcheck...")
    try:
        input_check = torch.randn(10, 5, dtype=torch.double, requires_grad=True)
        weight_check = torch.randn(8, 5, dtype=torch.double, requires_grad=True)

        result = gradcheck(
            MyCustomFunction.apply,
            (input_check, weight_check),
            eps=1e-6,
            atol=1e-4,
            raise_exception=True
        )

        print(f"   ✅ gradcheck PASSED!")
        print(f"\n🎉 All checks passed! Function is correct.")

    except RuntimeError as e:
        print(f"   ❌ gradcheck FAILED")
        print(f"   Error: {e}")
        print(f"\n   Debug hints:")
        print(f"   - Check gradient computation formulas")
        print(f"   - Verify all transposes are correct")
        print(f"   - Ensure shapes match everywhere")
        print(f"   - Check for in-place operations")
        print(f"   - Verify saved tensors are correct")

# Run debugging
debug_custom_function()
```

### Common Error Messages and Solutions

```python
"""
ERROR 1: "one of the variables needed for gradient computation has been modified by an inplace operation"

CAUSE: In-place operation in forward or backward

SOLUTION: Replace in-place ops with non-in-place versions
  - Replace: tensor[mask] = value
  - With: tensor = tensor.clone(); tensor[mask] = value
  - Or use: tensor * mask instead of masking


ERROR 2: "grad can be implicitly created only for scalar outputs"

CAUSE: Calling .backward() on non-scalar tensor without grad_output

SOLUTION: Either sum to scalar or provide grad_output
  - loss = output.sum(); loss.backward()
  - Or: output.backward(torch.ones_like(output))


ERROR 3: "Expected to get X gradient(s) for backward, but got Y"

CAUSE: backward() returns wrong number of gradients

SOLUTION: Return one gradient per forward() argument (excluding ctx)
  - forward(ctx, a, b, c) → backward must return (grad_a, grad_b, grad_c)
  - Use None for arguments that don't need gradients


ERROR 4: "Sizes of tensors must match except in dimension X"

CAUSE: Shape mismatch in gradient computation

SOLUTION: Ensure grad_input.shape == input.shape
  - Print shapes to debug: print(f"grad shape: {grad.shape}, input shape: {input.shape}")
  - Handle broadcasting by summing over broadcasted dimensions


ERROR 5: "RuntimeError: Function returned an invalid gradient at index X - got ... but expected shape ..."

CAUSE: Gradient shape doesn't match input shape

SOLUTION: Verify gradient shape matches input exactly
  - assert grad_input.shape == input.shape
  - Use .view(), .reshape(), .expand() to fix shape


ERROR 6: "gradcheck failed"

CAUSE: Analytical gradient ≠ numerical gradient

SOLUTION: Debug gradient computation
  - Check math formulas (derivatives)
  - Verify transposes in matrix operations
  - Test manually with small tensors
  - Run debug_custom_function() above


ERROR 7: "AttributeError: 'Context' object has no attribute 'saved_tensors'"

CAUSE: Accessing saved_tensors in forward (only available in backward)

SOLUTION: Only access saved_tensors in backward()
  - forward: ctx.save_for_backward(...)
  - backward: ctx.saved_tensors


ERROR 8: "TypeError: save_for_backward() takes 1 positional argument but X were given"

CAUSE: Passing non-tensors to save_for_backward()

SOLUTION: Only save tensors with save_for_backward(), use attributes for others
  - ctx.save_for_backward(tensor1, tensor2)  # Tensors only
  - ctx.some_value = non_tensor_data  # Non-tensors as attributes
"""
```

## When NOT to Use Custom Functions

Recognize when custom functions are unnecessary:

```python
# DON'T: Wrapping simple PyTorch operations
class UnnecessaryAdd(Function):  # ❌ Pointless
    @staticmethod
    def forward(ctx, a, b):
        return a + b
    @staticmethod
    def backward(ctx, grad):
        return grad, grad

# DO: Use PyTorch directly
output = a + b  # ✅ Autograd handles this


# DON'T: Reimplementing existing activations
class UnnecessaryReLU(Function):  # ❌ Use torch.relu
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input > 0).float()

# DO: Use built-in
output = torch.relu(input)  # ✅ Optimized C++ implementation


# DON'T: Operations that compose from standard ops
class UnnecessaryGELU(Function):  # ❌ Can compose from existing ops
    @staticmethod
    def forward(ctx, input):
        # GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # This is composed entirely from standard ops
        ctx.save_for_backward(input)
        return 0.5 * input * (1 + torch.tanh(...))
    @staticmethod
    def backward(ctx, grad_output):
        # Complex gradient computation
        ...

# DO: Let autograd handle composition
def gelu(input):  # ✅ Autograd computes gradients automatically
    return 0.5 * input * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (input + 0.044715 * input ** 3)
    ))

# Or even better: use built-in
output = torch.nn.functional.gelu(input)  # ✅ Most efficient


# WHEN TO USE: True custom operations
class ClippedGradientLinear(Function):  # ✅ Custom gradient behavior
    """
    Linear operation with clipped gradients.
    Can't compose from standard ops - requires custom backward.
    """
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return input @ weight.t()

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        # Custom: clip gradients (can't do with composition)
        grad_input = torch.clamp(grad_output @ weight, -1, 1)
        grad_weight = grad_output.t() @ input

        return grad_input, grad_weight
```

## gradcheck Time Investment

**Reality check on "no time for gradcheck"**:

When under deadline pressure, this rationalization is powerful but wrong:

```python
# Time cost analysis
gradcheck_time = "< 1 second"  # Typical operation
debugging_time = "hours to days"  # Finding gradient bugs without gradcheck

# ROI calculation
time_investment = 1  # second
time_saved = 3600 * 24  # potentially days
roi = time_saved / time_investment  # 86,400x return!
```

**Under deadline pressure, correct workflow**:
1. Implement Function correctly (5-10 minutes)
2. Run gradcheck (1 second)
3. Deploy with confidence

**NOT**:
1. Skip verification (saves 1 second)
2. Deploy immediately
3. Spend days debugging gradient bugs in production (costs hours/days)

**Time spent on gradcheck**: <1 second (negligible)
**Time saved from catching bugs early**: hours to days (enormous)

**The math is clear**: Always run gradcheck, even under extreme time pressure.

## Multiple Custom Functions Workflow

**When implementing multiple functions**:

✅ **CORRECT: One-at-a-time**
1. Implement Function 1
2. Test Function 1 with gradcheck
3. Verify Function 1 passes
4. Move to Function 2
5. Repeat for each function

❌ **WRONG: Batch approach**
1. Implement all functions
2. Test all together
3. Debug mess of overlapping bugs
4. Waste hours figuring out which function is broken
5. Fix bugs one at a time anyway (should have started here)

**Why one-at-a-time wins**:
- Bugs caught immediately (when you wrote the code, easy to debug)
- Know each function works before building on it
- Same total time, but bugs isolated (no "which function broke?" confusion)
- Build confidence incrementally
- Can use earlier functions while implementing later ones

**Example**:
```python
# Implementing 5 activations

# ✅ CORRECT
SwishBeta() → test → ✅ → Mish() → test → ✅ → GELU() → test → ✅ ...
# Each bug found immediately after writing that function

# ❌ WRONG
SwishBeta() + Mish() + GELU() + ELU() + SELU() → test all
# Bug found in one, but which? Have to debug all to find it
# 5x the debugging complexity
```

## Handling Approximate Gradients

**Special case**: External libraries with approximate gradients (finite differences, Monte Carlo estimates).

When wrapping external code with approximate gradients:

### Workflow for Approximate Gradients

```python
import torch
from torch.autograd import Function

class ApproximateGradientWrapper(Function):
    """
    Wrap external library with approximate gradients.

    Key insights:
    1. Standard gradcheck WILL fail (approximate ≠ analytical)
    2. But can still verify wrapper implementation
    3. Must quantify gradient quality
    4. Must assess if quality is acceptable
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return external_library_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Library provides approximate gradient
        grad_input = external_library_gradient(input, grad_output)
        return grad_input

# Can't run standard gradcheck, but CAN verify:

def test_approximate_gradient_wrapper():
    """Verification workflow for approximate gradients."""

    # 1. Verify wrapper mechanics (forward/backward run)
    input = torch.randn(5, requires_grad=True)
    output = ApproximateGradientWrapper.apply(input)
    output.sum().backward()
    assert input.grad is not None
    print("✅ Wrapper mechanics work")

    # 2. Quantify gradient quality (compare with numerical)
    input_test = torch.randn(3, requires_grad=True)
    output = ApproximateGradientWrapper.apply(input_test)
    output.sum().backward()
    library_gradient = input_test.grad.clone()

    # Compute our own numerical gradient
    eps = 1e-6
    numerical_gradient = compute_numerical_gradient(input_test, eps)

    # Measure error
    error = (library_gradient - numerical_gradient).abs().max()
    print(f"Gradient error: {error:.6e}")

    # 3. Assess acceptability
    if error < 1e-3:
        print("✅ Approximate gradient high quality (good enough)")
    elif error < 1e-2:
        print("⚠️  Approximate gradient has noticeable error (may affect optimization)")
    else:
        print("❌ Approximate gradient poor quality (likely to cause issues)")
        raise ValueError("Gradient quality unacceptable")

    # 4. Document in code
    print("\n📝 Documentation:")
    print(f"   - Gradients are approximate (error: {error:.6e})")
    print("   - Standard gradcheck will fail (expected)")
    print("   - Wrapper implementation verified correct")
    print("   - Gradient quality measured and acceptable")

    return error

# Run verification
gradient_error = test_approximate_gradient_wrapper()
```

### Key Points for Approximate Gradients

1. **Standard gradcheck will fail** - This is expected (approximate ≠ analytical)
2. **Test wrapper implementation** - Verify mechanics (forward/backward run)
3. **Quantify gradient quality** - Measure error vs numerical gradient
4. **Assess acceptability** - Is error tolerable for your use case?
5. **Document limitations** - Record gradient error magnitude in code/docs

**Don't skip verification** - Adapt it to approximate case.

**"Good enough" requires evidence** - Measure error, don't assume.

## Rationalization Resistance

| Rationalization | Reality | Counter-Response |
|----------------|---------|------------------|
| "Autograd will figure it out" | Only for standard ops; custom ops need Function | Use Function for non-standard operations. Autograd needs explicit backward implementation. |
| "Gradient looks mathematically correct" | Implementation bugs invisible without testing | Always run gradcheck. Math correctness ≠ implementation correctness. |
| "gradcheck is slow, skip for speed" | Catches bugs early; debugging later costs more | gradcheck is fast (<1s). Finding gradient bugs without it takes hours/days. |
| "No time for gradcheck, deadline NOW" | gradcheck takes <1s, debugging takes hours | 1 second now saves hours of production debugging. Always run gradcheck. |
| "Too complex for me" | Pattern is standardized; template works | Follow template. Thousands of successful implementations exist. |
| "In-place is more efficient" | Breaks autograd graph; causes crashes | Never use in-place in custom functions. Memory savings negligible, bugs catastrophic. |
| "Shape will probably work out" | Must match exactly; no flexibility | Gradient shape must equal input shape exactly. Verify with assertions. |
| "ctx details don't matter much" | Incorrect usage breaks everything | ctx.save_for_backward() is mandatory for tensors. Attributes break memory tracking. |
| "My manual test is good enough" | Misses edge cases gradcheck catches | Manual tests catch obvious bugs. gradcheck catches subtle numerical errors. |
| "Batch test all functions together" | Overlapping bugs hard to debug | Test one at a time. Bugs isolated immediately, same total time. |
| "Don't need needs_input_grad check" | Wastes computation; slower training | Always check needs_input_grad. Free optimization, no downside. |
| "Approximate gradients, can't verify" | Can verify wrapper and measure quality | Adapt verification. Test wrapper mechanics, quantify error, assess acceptability. |
| "Second-order derivatives too advanced" | Same pattern as first-order | Not advanced. Same template, test with gradgradcheck. Accessible to all. |
| "Can skip double backward support" | Breaks higher-order derivatives if needed | If you might need Hessian/meta-learning, support double backward from start. |
| "Detach doesn't matter here" | Controls gradient flow; critical | Understand when to detach. Impacts what gets gradients. |
| "I'll verify gradients during training" | Training metrics hide gradient bugs | Verify before training. Gradient bugs cause subtle issues (slow convergence, wrong behavior). |
| "Test in production, faster iteration" | Production debugging catastrophic | Test before deployment. Production gradient bugs cause model failure. |

## Red Flags Checklist

**Stop and verify if you see:**

1. ⚠️ **Not using torch.autograd.Function** for custom operations
   - Writing normal functions for truly custom ops (external code, novel math)

2. ⚠️ **No gradcheck before using** the function
   - Skipping numerical verification
   - "Testing during training" instead of proper gradcheck

3. ⚠️ **In-place operations** in forward or backward
   - `tensor[mask] = value`, `tensor += other`, `tensor.mul_(other)`
   - Modifying `grad_output` in backward

4. ⚠️ **Wrong number of return values** from backward
   - Not returning gradient for each forward() input
   - Missing None for non-tensor arguments

5. ⚠️ **Not using ctx.save_for_backward()** for tensors
   - Saving tensors as `ctx.tensor = tensor` instead
   - Saving non-tensors with save_for_backward()

6. ⚠️ **Gradient shape doesn't match** input shape
   - Not verifying `grad_input.shape == input.shape`
   - Forgetting to expand/sum for broadcasting

7. ⚠️ **Missing needs_input_grad checks**
   - Always computing all gradients even when not needed
   - Not checking `ctx.needs_input_grad[i]` before expensive computation

8. ⚠️ **Using .data instead of .detach()**
   - Accessing `.data` attribute
   - Not understanding difference between .data and .detach()

9. ⚠️ **Accessing ctx.saved_tensors in forward**
   - Trying to use saved tensors before backward
   - Not understanding ctx lifecycle

10. ⚠️ **Modifying saved tensors** in backward
    - In-place operations on tensors from ctx.saved_tensors
    - Breaking gradient graph

11. ⚠️ **Ignoring gradcheck failures**
    - "Gradient close enough"
    - Not investigating why gradcheck failed

12. ⚠️ **Using custom Function for standard operations**
    - Reimplementing built-in operations unnecessarily
    - Not checking if PyTorch already provides it

13. ⚠️ **Batching implementation without incremental testing**
    - Implementing multiple functions before testing any
    - "Will test them all together" approach

14. ⚠️ **Skipping gradcheck under time pressure**
    - "Deadline is tight, verify later"
    - "No time for gradcheck"

15. ⚠️ **Assuming approximate gradients can't be verified**
    - "Library provides gradients, can't test"
    - Not measuring gradient quality

16. ⚠️ **Avoiding second-order derivatives due to perceived complexity**
    - "Too advanced for me"
    - Not attempting gradgradcheck

17. ⚠️ **Deploying to production without verification**
    - "Test with real data"
    - Skipping numerical verification

## Summary Checklist

Before deploying a custom autograd function:

- [ ] Verified custom Function is actually needed (can't compose from standard ops)
- [ ] Implemented forward() correctly (saves needed tensors/data)
- [ ] Used ctx.save_for_backward() for ALL tensors
- [ ] Saved non-tensor data as ctx attributes
- [ ] Implemented backward() with correct gradient formulas
- [ ] Verified gradient shape matches input shape exactly
- [ ] Returned gradient for EVERY forward() input (None if not needed)
- [ ] Added needs_input_grad checks for expensive computations
- [ ] Avoided ALL in-place operations
- [ ] **Ran gradcheck and verified it PASSES**
- [ ] Tested with different input shapes
- [ ] Tested with optional parameters (if any)
- [ ] Verified no NaN/Inf in outputs or gradients
- [ ] Checked gradient magnitudes are reasonable
- [ ] Tested double backward if needed
- [ ] Added documentation explaining when to use this function
- [ ] Considered memory efficiency (detach, selective saving)

## Final Notes

**Custom autograd functions are powerful but require precision**:

1. Use them when truly needed (custom ops, external code, memory optimization, custom gradients)
2. Follow the template pattern (don't reinvent)
3. **Always run gradcheck** (non-negotiable)
4. Understand ctx rules (save_for_backward for tensors, attributes for non-tensors)
5. Verify shapes (gradient must match input)
6. Avoid in-place operations (they break autograd)
7. Check needs_input_grad (optimization)
8. Test thoroughly before using in training

**The gradient implementation is as important as the forward computation.** Bugs in backward() are silent and catastrophic - they cause models to learn wrong things. gradcheck is your safety net; never skip it.

When in doubt: implement, run gradcheck, debug until it passes, then use with confidence.
