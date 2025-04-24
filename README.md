# cpx-mat2by2

To power the rust-quantum project, this library provides a robust `AltMat` type for a single qubit, specifically optimized for the intensive complex number multiplications required for quantum gate applications and tensor product operations.

## Key Features

* **`AltMat` Type:** A versatile enum that can represent various types of 2x2 complex matrices relevant to quantum computing:
    * Rank-0 (Zero matrix)
    * Rank-1 Projectors
    * Rank-1 Nilpotent matrices
    * General Rank-1 matrices
    * Pauli matrices (I, X, Y, Z) with complex scaling
    * General Rank-2 matrices (via Singular Value Decomposition)
* **Optimized for Quantum Operations:** Designed with the performance requirements of quantum gate applications in mind, focusing on efficient complex number multiplication.
* **Integration with `cpx-coords`:** Leverages the `cpx-coords` crate for robust and accurate complex number arithmetic.
* **Core Quantum Concepts:** Provides structures and methods for representing quantum states (`State`), projectors, and related linear algebra concepts.
* **Essential Matrix Operations:** Includes methods for regularization, normalization, matrix conversion, trace calculation, and more.
* **Hashing:** Implements the `Hash` trait for all core types, enabling their use in data structures like hash maps and sets.

## Usage

### Installation

Add `cpx-mat2by2` as a dependency to your `Cargo.toml` file:

```toml
[dependencies]
cpx-mat2by2 = "0.1.0" # Replace with the latest version
```

## License

This project is licensed under either of

- [MIT license](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.


### Contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you shall be licensed as above, without any additional terms or conditions.

# cpx-mat2by2
