# cpx-mat2by2

**`cpx-mat2by2`** is a Rust library for representing **separable quantum states** and **structured 2×2 complex operators** in quantum computation.  
It is designed to integrate seamlessly with the [`cpx-coords`](https://crates.io/crates/cpx-coords) crate, which provides coordinate-system-aware complex number types `Cpx<T>`, where `T` is typically `f32` or `f64`.

This crate emphasizes **canonical normalization** of quantum states and operators, making equality checks robust, particularly useful when constructing entangled states or modeling non-trivial *k*-local operations.

---

## Design Principles

- **Global phase is ignored** in separable states and 2×2 matrices: canonicalization ensures the unique representation, which is only restored when coherent superpositions are explicitly handled.

- **Normalization is enforced** to ensure:
  - Pure states have unit norm and fixed global phase (`c₀ ∈ ℝ₊`)
  - Operators follow structured forms (e.g., unit determinant for rank-2 matrices)

- **Matrix classification:**
  - **Rank-0**: Zero matrix
  - **Rank-1**: Further subdivided into:
    - Projector: `|ψ⟩⟨ψ|`
    - Nilpotent: `|ψ⟩⟨ψ⊥|` or `|ψ⊥⟩⟨ψ|`
    - General outer product: `|ket⟩⟨bra|`
  - **Rank-2**: Full-rank 2×2 matrices, restricted to **unit determinant** and classified by Hermiticity

- **Pauli-based representations** (complex quaternion form) are used to encode all 2×2 operators, supporting both real (Hermitian) and complex coefficients.

- **Multi-qubit structures** use `BTreeMap<usize, _>` (not `HashMap`) to ensure deterministic key ordering.  
  This facilitates consistent hashing and composability for high-level quantum operations in future crates.


This crate is part of the broader rust-quantum project, focused on efficient numerical quantum computation.

## Usage

### Installation

Add `cpx-mat2by2` as a dependency to your `Cargo.toml` file:

```toml
[dependencies]
cpx-mat2by2 = "0.1.3" # Replace with the latest version
```

## License

This project is licensed under either of

- [MIT license](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.


### Contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you shall be licensed as above, without any additional terms or conditions.
