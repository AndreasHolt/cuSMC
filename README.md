<p align="center">
  <picture>
  <img src="./readme/cuSMC_light.png" width="330" alt="cuSMC logo">
</picture>

</p>

<h3 align="center">A CUDA-accelerated Statistical Model Checker for Stochastic Timed Automata</h3>


## Overview

cuSMC implements GPU-accelerated statistical model checking for stochastic timed automata. The tool parallelizes simulations through:

- Mapping model components to CUDA threads within blocks representing simulation runs
  - Each thread is responsible for its component's entire behavior including:
    - Evaluating guards on outgoing transitions
    - Selecting and taking enabled transitions
    - Sampling delays 
    - Updating component state and synchronizing with other components
- Using shared memory for efficient component state management, sampling  and synchronization between threads
- Maintaining an optimized model structure in global memory with:
    - Array-of-structs design for node, edge, guard, invariant and update data with cache-friendly access patterns
    - Contiguous memory layout for components at each network level
    - Read-only data caching through restrict and const qualifiers
    - Use of __ldg instruction for improved global memory loads of expression data
- Supporting broadcast synchronization through shared memory
- Thread-level sampling and component race resolution through:
    - Per-thread uniform distribution sampling for bounded delays
    - Per-thread exponential distribution sampling for unbounded delays
    - O(log n) parallel reduction to determine race winners

## Acknowledgements

cuSMC utilises [SMACC's](https://github.com/Baksling/P7-SMAcc) UPPAAL XML parser. We also utilise their structure for expressions and their helper functions to evaluate non-PN and PN expressions.

## Requirements

- C++17 compiler
- CUDA Toolkit (≥11.0)
- CMake (≥3.18)

## Building

```bash
mkdir build
cd build
cmake ..
make
```



