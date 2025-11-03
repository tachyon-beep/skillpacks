---
name: scientific-computing-foundations
description: NumPy/pandas mastery, vectorization, memory efficiency, performance anti-patterns, data pipeline patterns, large datasets, array type hints
---

# Scientific Computing Foundations

## Overview

**Core Principle:** Vectorize operations, avoid loops. NumPy and pandas are built on C/Fortran code that's orders of magnitude faster than Python loops. The biggest performance gains come from eliminating iteration over rows/elements.

Scientific computing in Python centers on NumPy (arrays) and pandas (dataframes). These libraries enable fast numerical computation on large datasets through vectorized operations and efficient memory layouts. The most common mistake: using Python loops when vectorized operations exist.

## When to Use

**Use this skill when:**
- "NumPy operations"
- "Pandas DataFrame slow"
- "Vectorization"
- "How to avoid loops?"
- "DataFrame iteration"
- "Array performance"
- "Memory usage too high"
- "Large dataset processing"

**Don't use when:**
- Setting up project (use project-structure-and-tooling)
- Profiling needed first (use debugging-and-profiling)
- ML pipeline orchestration (use ml-engineering-workflows)

**Symptoms triggering this skill:**
- Slow DataFrame operations
- High memory usage with arrays
- Using loops over DataFrame rows
- Need to process large datasets efficiently

---

## NumPy Fundamentals

### Array Creation and Types

```python
import numpy as np

# ❌ WRONG: Creating arrays from Python lists in loop
data = []
for i in range(1000000):
    data.append(i * 2)
arr = np.array(data)

# ✅ CORRECT: Use NumPy functions
arr = np.arange(1000000) * 2

# ✅ CORRECT: Pre-allocate for known size
arr = np.empty(1000000, dtype=np.int64)
for i in range(1000000):
    arr[i] = i * 2  # Still slow, but better than list

# ✅ BETTER: Fully vectorized
arr = np.arange(1000000, dtype=np.int64) * 2

# ✅ CORRECT: Specify dtype for memory efficiency
# float64 (default): 8 bytes per element
# float32: 4 bytes per element
large_arr = np.zeros(1000000, dtype=np.float32)  # Half the memory

# Why this matters: dtype affects both memory usage and performance
# Use smallest dtype that fits your data
```

### Vectorized Operations

```python
# ❌ WRONG: Loop over array elements
arr = np.arange(1000000)
result = np.empty(1000000)
for i in range(len(arr)):
    result[i] = arr[i] ** 2 + 2 * arr[i] + 1

# ✅ CORRECT: Vectorized operations
arr = np.arange(1000000)
result = arr ** 2 + 2 * arr + 1

# Speed difference: ~100x faster with vectorization

# ❌ WRONG: Element-wise comparison in loop
matches = []
for val in arr:
    if val > 100:
        matches.append(val)
result = np.array(matches)

# ✅ CORRECT: Boolean indexing
result = arr[arr > 100]

# ✅ CORRECT: Complex conditions
result = arr[(arr > 100) & (arr < 200)]  # Note: & not 'and'
result = arr[(arr < 50) | (arr > 150)]   # Note: | not 'or'
```

**Why this matters**: Vectorized operations run in C, avoiding Python interpreter overhead. 10-100x speedup typical.

### Broadcasting

```python
# Broadcasting: Operating on arrays of different shapes

# ✅ CORRECT: Scalar broadcasting
arr = np.array([1, 2, 3, 4])
result = arr + 10  # [11, 12, 13, 14]

# ✅ CORRECT: 1D array broadcast to 2D
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

row_vector = np.array([10, 20, 30])
result = matrix + row_vector
# [[11, 22, 33],
#  [14, 25, 36],
#  [17, 28, 39]]

# ✅ CORRECT: Column vector broadcasting
col_vector = np.array([[10],
                       [20],
                       [30]])
result = matrix + col_vector
# [[11, 12, 13],
#  [24, 25, 26],
#  [37, 38, 39]]

# ✅ CORRECT: Add axis for broadcasting
row = np.array([1, 2, 3])
col = row[:, np.newaxis]  # Convert to column vector
# col shape: (3, 1)

# Outer product via broadcasting
outer = row[np.newaxis, :] * col
# [[1, 2, 3],
#  [2, 4, 6],
#  [3, 6, 9]]

# ❌ WRONG: Manual broadcasting with loops
result = np.empty_like(matrix)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        result[i, j] = matrix[i, j] + row_vector[j]

# Why this matters: Broadcasting eliminates loops and is much faster
```

### Memory-Efficient Operations

```python
# ❌ WRONG: Creating unnecessary copies
large_arr = np.random.rand(10000, 10000)  # ~800MB
result1 = large_arr + 1  # Creates new 800MB array
result2 = result1 * 2    # Creates another 800MB array
# Total: 2.4GB memory usage

# ✅ CORRECT: In-place operations
large_arr = np.random.rand(10000, 10000)
large_arr += 1  # Modifies in-place, no copy
large_arr *= 2  # Modifies in-place, no copy
# Total: 800MB memory usage

# ✅ CORRECT: Use 'out' parameter
result = np.empty_like(large_arr)
np.add(large_arr, 1, out=result)
np.multiply(result, 2, out=result)

# ❌ WRONG: Unnecessary array copies
arr = np.arange(1000000)
subset = arr[::2].copy()  # Explicit copy needed? Check first
subset[0] = 999  # Doesn't affect arr

# ✅ CORRECT: Views avoid copies (when possible)
arr = np.arange(1000000)
view = arr[::2]  # View, not copy (shares memory)
view[0] = 999  # Modifies arr too!

# Check if view or copy
print(arr.base is None)  # False = view, True = owns memory
```

**Why this matters**: Large arrays consume lots of memory. In-place operations and views avoid copies, reducing memory usage significantly.

### Aggregations and Reductions

```python
# ✅ CORRECT: Axis-aware aggregations
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Sum all elements
total = matrix.sum()  # 45

# Sum along axis 0 (columns)
col_sums = matrix.sum(axis=0)  # [12, 15, 18]

# Sum along axis 1 (rows)
row_sums = matrix.sum(axis=1)  # [6, 15, 24]

# ❌ WRONG: Manual aggregation
total = 0
for row in matrix:
    for val in row:
        total += val

# ✅ CORRECT: Multiple aggregations
matrix.mean()
matrix.std()
matrix.min()
matrix.max()
matrix.argmin()  # Index of minimum
matrix.argmax()  # Index of maximum

# ✅ CORRECT: Conditional aggregations
# Sum only positive values
positive_sum = matrix[matrix > 0].sum()

# Count elements > 5
count = (matrix > 5).sum()

# Percentage > 5
percentage = (matrix > 5).mean() * 100
```

---

## pandas Fundamentals

### DataFrame Creation

```python
import pandas as pd

# ❌ WRONG: Building DataFrame row by row
df = pd.DataFrame()
for i in range(10000):
    df = pd.concat([df, pd.DataFrame({'a': [i], 'b': [i*2]})], ignore_index=True)
# Extremely slow: O(n²) complexity

# ✅ CORRECT: Create from dict of lists
data = {
    'a': list(range(10000)),
    'b': [i * 2 for i in range(10000)]
}
df = pd.DataFrame(data)

# ✅ BETTER: Use NumPy arrays
df = pd.DataFrame({
    'a': np.arange(10000),
    'b': np.arange(10000) * 2
})

# ✅ CORRECT: From records
records = [{'a': i, 'b': i*2} for i in range(10000)]
df = pd.DataFrame.from_records(records)
```

### The Iteration Anti-Pattern

```python
# ❌ WRONG: iterrows() - THE MOST COMMON MISTAKE
df = pd.DataFrame({
    'value': np.random.rand(100000),
    'category': np.random.choice(['A', 'B', 'C'], 100000)
})

result = []
for idx, row in df.iterrows():  # VERY SLOW
    if row['value'] > 0.5:
        result.append(row['value'] * 2)

# ✅ CORRECT: Vectorized operations
mask = df['value'] > 0.5
result = df.loc[mask, 'value'] * 2

# Speed difference: ~100x faster

# ❌ WRONG: apply() on axis=1 (still row-by-row)
df['result'] = df.apply(
    lambda row: row['value'] * 2 if row['value'] > 0.5 else 0,
    axis=1
)
# Still slow: applies Python function to each row

# ✅ CORRECT: Vectorized with np.where
df['result'] = np.where(df['value'] > 0.5, df['value'] * 2, 0)

# ✅ CORRECT: Boolean indexing + assignment
df['result'] = 0
df.loc[df['value'] > 0.5, 'result'] = df['value'] * 2
```

**Why this matters**: `iterrows()` is the single biggest DataFrame performance killer. ALWAYS look for vectorized alternatives.

### Efficient Filtering and Selection

```python
df = pd.DataFrame({
    'A': np.random.rand(100000),
    'B': np.random.rand(100000),
    'C': np.random.choice(['X', 'Y', 'Z'], 100000)
})

# ❌ WRONG: Chaining filters inefficiently
df_filtered = df[df['A'] > 0.5]
df_filtered = df_filtered[df_filtered['B'] < 0.3]
df_filtered = df_filtered[df_filtered['C'] == 'X']

# ✅ CORRECT: Single boolean mask
mask = (df['A'] > 0.5) & (df['B'] < 0.3) & (df['C'] == 'X')
df_filtered = df[mask]

# ✅ CORRECT: query() for complex filters (cleaner syntax)
df_filtered = df.query('A > 0.5 and B < 0.3 and C == "X"')

# ✅ CORRECT: isin() for multiple values
df_filtered = df[df['C'].isin(['X', 'Y'])]

# ❌ WRONG: String matching in loop
matches = []
for val in df['C']:
    if 'X' in val:
        matches.append(True)
    else:
        matches.append(False)
df_filtered = df[matches]

# ✅ CORRECT: Vectorized string operations
df_filtered = df[df['C'].str.contains('X')]
```

### GroupBy Operations

```python
df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], 100000),
    'value': np.random.rand(100000),
    'count': np.random.randint(1, 100, 100000)
})

# ❌ WRONG: Manual grouping
groups = {}
for idx, row in df.iterrows():
    cat = row['category']
    if cat not in groups:
        groups[cat] = []
    groups[cat].append(row['value'])

results = {cat: sum(vals) / len(vals) for cat, vals in groups.items()}

# ✅ CORRECT: GroupBy
results = df.groupby('category')['value'].mean()

# ✅ CORRECT: Multiple aggregations
results = df.groupby('category').agg({
    'value': ['mean', 'std', 'min', 'max'],
    'count': 'sum'
})

# ✅ CORRECT: Named aggregations (pandas 0.25+)
results = df.groupby('category').agg(
    mean_value=('value', 'mean'),
    std_value=('value', 'std'),
    total_count=('count', 'sum')
)

# ✅ CORRECT: Custom aggregation function
def range_func(x):
    return x.max() - x.min()

results = df.groupby('category')['value'].agg(range_func)

# ✅ CORRECT: Transform (keeps original shape)
df['value_centered'] = df.groupby('category')['value'].transform(
    lambda x: x - x.mean()
)
```

**Why this matters**: GroupBy is highly optimized. Much faster than manual grouping. Use built-in aggregations when possible.

---

## Performance Anti-Patterns

### Anti-Pattern 1: DataFrame Iteration

```python
# ❌ WRONG: Iterating over rows
for idx, row in df.iterrows():
    df.at[idx, 'new_col'] = row['a'] + row['b']

# ✅ CORRECT: Vectorized column operation
df['new_col'] = df['a'] + df['b']

# ❌ WRONG: Itertuples (better than iterrows, but still slow)
for row in df.itertuples():
    # Process row...

# ✅ CORRECT: Use vectorized operations or apply to columns
```

### Anti-Pattern 2: Repeated Concatenation

```python
# ❌ WRONG: Growing DataFrame in loop
df = pd.DataFrame()
for i in range(10000):
    df = pd.concat([df, new_row_df], ignore_index=True)
# O(n²) complexity, extremely slow

# ✅ CORRECT: Collect data, then create DataFrame
data = []
for i in range(10000):
    data.append({'a': i, 'b': i*2})
df = pd.DataFrame(data)

# ✅ CORRECT: Pre-allocate NumPy array
arr = np.empty((10000, 2))
for i in range(10000):
    arr[i] = [i, i*2]
df = pd.DataFrame(arr, columns=['a', 'b'])
```

### Anti-Pattern 3: Using apply When Vectorized Exists

```python
# ❌ WRONG: apply() for simple operations
df['result'] = df['value'].apply(lambda x: x * 2)

# ✅ CORRECT: Direct vectorized operation
df['result'] = df['value'] * 2

# ❌ WRONG: apply() for conditions
df['category'] = df['value'].apply(lambda x: 'high' if x > 0.5 else 'low')

# ✅ CORRECT: np.where or pd.cut
df['category'] = np.where(df['value'] > 0.5, 'high', 'low')

# ✅ CORRECT: pd.cut for binning
df['category'] = pd.cut(df['value'], bins=[0, 0.5, 1.0], labels=['low', 'high'])

# When apply IS appropriate:
# - Complex logic not vectorizable
# - Need to call external function per row
# But verify vectorization truly impossible first
```

### Anti-Pattern 4: Not Using Categorical Data

```python
# ❌ WRONG: String columns for repeated values
df = pd.DataFrame({
    'category': ['A'] * 10000 + ['B'] * 10000 + ['C'] * 10000
})
# Memory: ~240KB (each string stored separately)

# ✅ CORRECT: Categorical type
df['category'] = pd.Categorical(df['category'])
# Memory: ~30KB (integers + small string table)

# ✅ CORRECT: Define categories at creation
df = pd.DataFrame({
    'category': pd.Categorical(
        ['A'] * 10000 + ['B'] * 10000,
        categories=['A', 'B', 'C']
    )
})

# When to use categorical:
# - Limited number of unique values (< 50% of rows)
# - Repeated string/object values
# - Memory constraints
# - Faster groupby operations
```

---

## Memory Optimization

### Choosing Appropriate dtypes

```python
# ❌ WRONG: Default dtypes waste memory
df = pd.DataFrame({
    'int_col': [1, 2, 3, 4, 5],  # int64 by default
    'float_col': [1.0, 2.0, 3.0],  # float64 by default
    'str_col': ['a', 'b', 'c', 'd', 'e']  # object dtype
})

print(df.memory_usage(deep=True))

# ✅ CORRECT: Optimize dtypes
df = pd.DataFrame({
    'int_col': pd.array([1, 2, 3, 4, 5], dtype='int8'),  # -128 to 127
    'float_col': pd.array([1.0, 2.0, 3.0], dtype='float32'),
    'str_col': pd.Categorical(['a', 'b', 'c', 'd', 'e'])
})

# ✅ CORRECT: Downcast after loading
df = pd.read_csv('data.csv')
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')

# Integer dtype ranges:
# int8: -128 to 127
# int16: -32,768 to 32,767
# int32: -2.1B to 2.1B
# int64: -9.2E18 to 9.2E18

# Float dtype precision:
# float16: ~3 decimal digits (rarely used)
# float32: ~7 decimal digits
# float64: ~15 decimal digits
```

### Chunked Processing for Large Files

```python
# ❌ WRONG: Loading entire file into memory
df = pd.read_csv('huge_file.csv')  # 10GB file, OOM!
df_processed = process_dataframe(df)

# ✅ CORRECT: Process in chunks
chunk_size = 100000
results = []

for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    processed = process_dataframe(chunk)
    results.append(processed)

df_final = pd.concat(results, ignore_index=True)

# ✅ CORRECT: Streaming aggregation
totals = {'A': 0, 'B': 0, 'C': 0}

for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    for col in totals:
        totals[col] += chunk[col].sum()

# ✅ CORRECT: Only load needed columns
df = pd.read_csv('huge_file.csv', usecols=['col1', 'col2', 'col3'])
```

### Using Sparse Data Structures

```python
# ❌ WRONG: Dense array for sparse data
# Data with 99% zeros
dense = np.zeros(1000000)
dense[::100] = 1  # Only 1% non-zero
# Memory: 8MB (float64 * 1M)

# ✅ CORRECT: Sparse array
from scipy.sparse import csr_matrix
sparse = csr_matrix(dense)
# Memory: ~80KB (only stores non-zero values + indices)

# ✅ CORRECT: Sparse DataFrame
df = pd.DataFrame({
    'A': pd.arrays.SparseArray([0] * 100 + [1] + [0] * 100),
    'B': pd.arrays.SparseArray([0] * 50 + [2] + [0] * 150)
})
```

---

## Data Pipeline Patterns

### Method Chaining

```python
# ❌ WRONG: Many intermediate variables
df = pd.read_csv('data.csv')
df = df[df['value'] > 0]
df = df.groupby('category')['value'].mean()
df = df.reset_index()
df = df.rename(columns={'value': 'mean_value'})

# ✅ CORRECT: Method chaining
df = (
    pd.read_csv('data.csv')
    .query('value > 0')
    .groupby('category')['value']
    .mean()
    .reset_index()
    .rename(columns={'value': 'mean_value'})
)

# ✅ CORRECT: Pipe for custom functions
def remove_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    return df[
        (df[column] > mean - n_std * std) &
        (df[column] < mean + n_std * std)
    ]

df = (
    pd.read_csv('data.csv')
    .pipe(remove_outliers, 'value', n_std=2)
    .groupby('category')['value']
    .mean()
)
```

### Efficient Merges and Joins

```python
# ❌ WRONG: Multiple small merges
for small_df in list_of_dfs:
    main_df = main_df.merge(small_df, on='key')
# Inefficient: creates many intermediate copies

# ✅ CORRECT: Merge all at once
df_merged = pd.concat(list_of_dfs, ignore_index=True)

# ✅ CORRECT: Optimize merge with sorted/indexed data
df1 = df1.set_index('key').sort_index()
df2 = df2.set_index('key').sort_index()
result = df1.merge(df2, left_index=True, right_index=True)

# ✅ CORRECT: Use indicator to track merge sources
result = df1.merge(df2, on='key', how='outer', indicator=True)
print(result['_merge'].value_counts())
# Shows: left_only, right_only, both

# ❌ WRONG: Cartesian product by accident
# df1: 1000 rows, df2: 1000 rows
result = df1.merge(df2, on='wrong_key')
# result: 1,000,000 rows! (if all keys match)

# ✅ CORRECT: Validate merge
result = df1.merge(df2, on='key', validate='1:1')
# Raises error if not one-to-one relationship
```

### Handling Missing Data

```python
# ❌ WRONG: Dropping all rows with any NaN
df_clean = df.dropna()  # Might lose most of data

# ✅ CORRECT: Drop rows with NaN in specific columns
df_clean = df.dropna(subset=['important_col1', 'important_col2'])

# ✅ CORRECT: Fill NaN with appropriate values
df['numeric_col'] = df['numeric_col'].fillna(df['numeric_col'].mean())
df['category_col'] = df['category_col'].fillna('Unknown')

# ✅ CORRECT: Forward/backward fill for time series
df['value'] = df['value'].fillna(method='ffill')  # Forward fill

# ✅ CORRECT: Interpolation
df['value'] = df['value'].interpolate(method='linear')

# ❌ WRONG: Not checking for NaN before operations
result = df['value'].mean()  # NaN propagates, might return NaN

# ✅ CORRECT: Explicit NaN handling
result = df['value'].mean(skipna=True)  # Default, but explicit is better
```

---

## Advanced NumPy Techniques

### Universal Functions (ufuncs)

```python
# ✅ CORRECT: Using built-in ufuncs
arr = np.random.rand(1000000)

# Trigonometric
result = np.sin(arr)
result = np.cos(arr)

# Exponential
result = np.exp(arr)
result = np.log(arr)

# Comparison
result = np.maximum(arr, 0.5)  # Element-wise max with scalar
result = np.minimum(arr, 0.5)

# ✅ CORRECT: Custom ufunc with @vectorize
from numba import vectorize

@vectorize
def custom_func(x):
    if x > 0.5:
        return x ** 2
    else:
        return x ** 3

result = custom_func(arr)  # Runs at C speed
```

### Advanced Indexing

```python
# ✅ CORRECT: Fancy indexing
arr = np.arange(100)
indices = [0, 5, 10, 15, 20]
result = arr[indices]  # Select specific indices

# ✅ CORRECT: Boolean indexing with multiple conditions
arr = np.random.rand(1000000)
mask = (arr > 0.3) & (arr < 0.7)
result = arr[mask]

# ✅ CORRECT: np.where for conditional replacement
arr = np.random.rand(1000)
result = np.where(arr > 0.5, arr, 0)  # Replace values <= 0.5 with 0

# ✅ CORRECT: Multi-dimensional indexing
matrix = np.random.rand(100, 100)
rows = [0, 10, 20]
cols = [5, 15, 25]
result = matrix[rows, cols]  # Select specific elements

# Get diagonal
diagonal = matrix[np.arange(100), np.arange(100)]
# Or use np.diag
diagonal = np.diag(matrix)
```

### Linear Algebra Operations

```python
# ✅ CORRECT: Matrix multiplication
A = np.random.rand(1000, 500)
B = np.random.rand(500, 200)
C = A @ B  # Python 3.5+ matrix multiply operator

# Or
C = np.dot(A, B)
C = np.matmul(A, B)

# ✅ CORRECT: Solve linear system Ax = b
A = np.random.rand(100, 100)
b = np.random.rand(100)
x = np.linalg.solve(A, b)

# ✅ CORRECT: Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# ✅ CORRECT: SVD (Singular Value Decomposition)
U, s, Vt = np.linalg.svd(A)

# ✅ CORRECT: Inverse
A_inv = np.linalg.inv(A)

# ❌ WRONG: Using inverse for solving Ax = b
x = np.linalg.inv(A) @ b  # Slower and less numerically stable

# ✅ CORRECT: Use solve directly
x = np.linalg.solve(A, b)
```

---

## Type Hints for NumPy and pandas

### NumPy Type Hints

```python
import numpy as np
from numpy.typing import NDArray

# ✅ CORRECT: Type hint for NumPy arrays
def process_array(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    return arr * 2

# ✅ CORRECT: Generic array type
def normalize(arr: NDArray) -> NDArray:
    return (arr - arr.mean()) / arr.std()

# ✅ CORRECT: Shape-specific type hints (Python 3.11+)
from typing import TypeAlias

Vector: TypeAlias = NDArray[np.float64]  # 1D array
Matrix: TypeAlias = NDArray[np.float64]  # 2D array

def matrix_multiply(A: Matrix, B: Matrix) -> Matrix:
    return A @ B
```

### pandas Type Hints

```python
import pandas as pd

# ✅ CORRECT: Type hints for Series and DataFrame
def process_series(s: pd.Series) -> pd.Series:
    return s * 2

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['value'] > 0]

# ✅ CORRECT: More specific DataFrame types (using Protocols)
from typing import Protocol

class DataFrameWithColumns(Protocol):
    """DataFrame with specific columns."""
    def __getitem__(self, key: str) -> pd.Series: ...

def analyze_data(df: DataFrameWithColumns) -> float:
    return df['value'].mean()
```

---

## Real-World Patterns

### Time Series Operations

```python
# ✅ CORRECT: Efficient time series resampling
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000000, freq='1s'),
    'value': np.random.rand(1000000)
})

df = df.set_index('timestamp')

# Resample to 1-minute intervals
df_resampled = df.resample('1min').agg({
    'value': ['mean', 'std', 'min', 'max']
})

# ✅ CORRECT: Rolling window operations
df['rolling_mean'] = df['value'].rolling(window=60).mean()
df['rolling_std'] = df['value'].rolling(window=60).std()

# ✅ CORRECT: Lag features
df['value_lag1'] = df['value'].shift(1)
df['value_lag60'] = df['value'].shift(60)

# ✅ CORRECT: Difference for stationarity
df['value_diff'] = df['value'].diff()
```

### Multi-Index Operations

```python
# ✅ CORRECT: Creating multi-index DataFrame
df = pd.DataFrame({
    'country': ['USA', 'USA', 'UK', 'UK'],
    'city': ['NYC', 'LA', 'London', 'Manchester'],
    'value': [100, 200, 150, 175]
})

df = df.set_index(['country', 'city'])

# Accessing with multi-index
df.loc['USA']  # All USA cities
df.loc[('USA', 'NYC')]  # Specific city

# ✅ CORRECT: Cross-section
df.xs('USA', level='country')
df.xs('London', level='city')

# ✅ CORRECT: GroupBy with multi-index
df.groupby(level='country').mean()
```

### Parallel Processing with Dask

```python
# For datasets larger than memory, use Dask (not in plan detail, but worth mentioning)
import dask.dataframe as dd

# ✅ CORRECT: Dask for out-of-core processing
df = dd.read_csv('huge_file.csv')
result = df.groupby('category')['value'].mean().compute()

# Dask uses same API as pandas, but lazy evaluation
# Only computes when .compute() is called
```

---

## Anti-Pattern Summary

### Top 5 Performance Killers

1. **iterrows()** - Use vectorized operations
2. **Growing DataFrame in loop** - Collect data, then create DataFrame
3. **apply() for simple operations** - Use vectorized alternatives
4. **Not using categorical for strings** - Convert to categorical
5. **Loading entire file when chunking works** - Use chunksize parameter

### Memory Usage Mistakes

1. **Using float64 when float32 sufficient** - Halves memory
2. **Not using categorical for repeated strings** - 10x memory savings
3. **Creating unnecessary copies** - Use in-place operations
4. **Loading all columns when few needed** - Use usecols parameter

---

## Decision Trees

### Should I Use NumPy or pandas?

```
What's your data structure?
├─ Homogeneous numeric array → NumPy
├─ Heterogeneous tabular data → pandas
├─ Time series → pandas
└─ Linear algebra → NumPy
```

### How to Optimize DataFrame Operation?

```
Can I vectorize?
├─ Yes → Use vectorized pandas/NumPy operations
└─ No → Can I use groupby?
    ├─ Yes → Use groupby with built-in aggregations
    └─ No → Can I use apply on columns (not rows)?
        ├─ Yes → Use apply on Series
        └─ No → Use itertuples (last resort)
```

### Memory Optimization Strategy

```
Is memory usage high?
├─ Yes → Check dtypes (downcast if possible)
│   └─ Still high? → Use categorical for strings
│       └─ Still high? → Process in chunks
└─ No → Continue with current approach
```

---

## Integration with Other Skills

**After using this skill:**
- If need ML pipelines → See @ml-engineering-workflows for experiment tracking
- If performance issues persist → See @debugging-and-profiling for profiling
- If type hints needed → See @modern-syntax-and-types for advanced typing

**Before using this skill:**
- If unsure if slow → Use @debugging-and-profiling to profile first
- If setting up project → Use @project-structure-and-tooling for dependencies

---

## Quick Reference

### NumPy Quick Wins

```python
# Vectorization
result = arr ** 2 + 2 * arr + 1  # Not: for loop

# Boolean indexing
result = arr[arr > 0]  # Not: list comprehension

# Broadcasting
result = matrix + row_vector  # Not: loop over rows

# In-place operations
arr += 1  # Not: arr = arr + 1
```

### pandas Quick Wins

```python
# Never iterrows
df['new'] = df['a'] + df['b']  # Not: iterrows

# Vectorized conditions
df['category'] = np.where(df['value'] > 0.5, 'high', 'low')

# Categorical for strings
df['category'] = pd.Categorical(df['category'])

# Query for complex filters
df.query('A > 0.5 and B < 0.3')  # Not: multiple []
```

### Memory Optimization Checklist

- [ ] Use smallest dtype that fits data
- [ ] Convert repeated strings to categorical
- [ ] Use chunking for files > available RAM
- [ ] Avoid unnecessary copies (use views or in-place ops)
- [ ] Only load needed columns (usecols in read_csv)
