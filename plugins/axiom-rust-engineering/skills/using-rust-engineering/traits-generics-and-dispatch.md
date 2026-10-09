# Rust trait and dispatch contracts

## Start with consumers

Identify which operations need static specialization, runtime heterogeneity or a public extension point. Choose generics, trait objects, enums or concrete types from those requirements rather than treating one dispatch form as universally superior.

## Check the boundary

- Object safety/dyn compatibility constrains receiver, generic methods, associated items and return types. Verify with the supported compiler and actual call site.
- Lifetimes on trait objects and returned references must match ownership. Adding `'static` can exclude valid borrowed consumers rather than solve the design.
- Blanket implementations can overlap or block future downstream implementations. Check coherence and public API evolution.
- Associated types bind a type per implementation; generic parameters permit different instantiations. Choose according to how callers select the relationship.
- Sealing a trait limits downstream implementations and is a product/API decision, not a default lint fix.
- Boxing/erasure affects allocation, Send bounds and cancellation behavior for async interfaces. Verify those obligations.
- Monomorphization can trade dispatch overhead for binary size/build time. Measure before claiming a benefit.
- Unsafe marker traits need an explicit implementor contract and verified soundness assumptions.

Compile representative downstream consumers and negative cases that define the API. Keep generic abstractions as small as the actual variation requires.
