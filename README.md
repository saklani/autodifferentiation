# OCaml Tensor

A lightweight tensor computation library implemented in OCaml with automatic differentiation support.

## Overview

This library provides a tensor manipulation system with two main components:

1. **Tensor Module**: Core tensor operations with support for n-dimensional arrays
2. **Variable Module**: Automatic differentiation with a computational graph implementation

The library is designed for machine learning applications, numerical computing, and gradient-based optimization.

## Features

### Tensor Operations

- N-dimensional array support with float values
- Element-wise operations (add, subtract, multiply, divide)
- Matrix operations (dot product, matrix multiplication, transpose)
- Shape manipulation (reshape, flatten, swapaxes)
- Broadcasting support for operations between tensors of different shapes
- Mathematical functions (exp, log, sin, cos, tan)
- Reduction operations (sum along axes)

### Automatic Differentiation

- Forward and backward gradient computation
- Computational graph tracking
- Support for machine learning operations:
  - Activation functions (sigmoid, leaky ReLU)
  - Loss functions (binary cross-entropy)
  - Softmax for classification problems
- Gradient visualization capabilities

## Installation

### Dependencies

- OCaml 4.13.0 or higher
- Core library
- Bigarray module

### Build with Dune

```bash
dune build
```

## Usage Examples

### Basic Tensor Operations

```ocaml
open Tensor

(* Create tensors *)
let a = zeros [|2; 3|]
let b = ones [|2; 3|]
let c = random [|2; 3|]

(* Element-wise operations *)
let d = a + b
let e = c * d
let f = log e

(* Matrix operations *)
let g = transpose c
let h = matmul c g
```

### Automatic Differentiation

```ocaml
open Variable

(* Create variables for computation *)
let x = random [|2; 2|]
let y = random [|2; 2|]

(* Build computation graph *)
let z = x * y + sin y

(* Compute gradients *)
let grads = gradients z
let dx = find grads x
let dy = find grads y

(* Print gradients *)
print_table grads
```

### Neural Network Components

```ocaml
open Variable

(* Data *)
let x_data = random [|10; 5|]  (* 10 samples, 5 features *)
let y_true = random [|10; 1|]  (* 10 samples, binary labels *)

(* Simple neural network with one hidden layer *)
let w1 = random [|5; 3|]       (* 5 input features, 3 hidden neurons *)
let b1 = zeros [|1; 3|]
let w2 = random [|3; 1|]       (* 3 hidden neurons, 1 output *)
let b2 = zeros [|1; 1|]

(* Forward pass *)
let hidden = leaky_relu (matmul x_data w1 + b1)
let y_pred = sigmoid (matmul hidden w2 + b2)

(* Loss computation *)
let loss = binary_cross_entropy y_true y_pred

(* Compute gradients *)
let grads = gradients loss
```

## Visualization

The library includes a visualization feature for computational graphs:

```ocaml
open Variable

let x = random [|2; 2|]
let y = random [|2; 2|]
let z = x * sin y

(* Output the computation graph to a DOT file *)
visualize z "computation_graph.dot"

(* Convert to PNG with GraphViz *)
(* system("dot -Tpng computation_graph.dot -o computation_graph.png") *)
```

## API Documentation

### Tensor Module

The `Tensor` module provides core tensor operations:

- Creation functions: `zeros`, `ones`, `random`, `create`
- Element-wise operations: `add`, `sub`, `mul`, `div`, `neg`
- Matrix operations: `dot`, `matmul`, `transpose`
- Shape operations: `reshape`, `flatten`, `swapaxes`
- Math functions: `log`, `exp`, `sin`, `cos`, `tan`, `pow`
- Reduction: `sum` (with optional axis)

### Variable Module

The `Variable` module builds on the `Tensor` module, adding automatic differentiation:

- Creation: `create`, `zero`, `one`, `zeros`, `random`
- Operations: `add`, `mul`, `div`, `sub`, `neg`, `inv`
- Math functions: `log`, `exp`, `sin`, `cos`, `tan`
- Gradient computation: `gradients`, `find`
- Machine learning functions: `sigmoid`, `leaky_relu`, `softmax`, `binary_cross_entropy`
- Visualization: `visualize`

## License

