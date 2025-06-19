//! The `cpx-mat2by2` library provides structures for representing and manipulating
//! quantum states and operators in a 2-dimensional Hilbert space (qubit).
//! It leverages the `cpx-coords` crate for complex number arithmetic.

use crate::BraKet::{Bra, Ket};
use crate::ProNil::{Nilpotent, Projector};
use core::cmp::{max, min};
use core::hash::Hash;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use cpx_coords::FloatExt;
use cpx_coords::*;
use std::collections::BTreeMap;

/// Enum representing the side of a state vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BraKet {
    /// A bra vector ⟨ψ|.
    Bra,
    /// A ket vector |ψ⟩.
    Ket,
}

/// Canonical normalized amplitudes representing a single-qubit pure state.
///
/// This form enforces uniqueness for hashing and equality by fixing the global phase:
/// - `c0 ∈ ℝ≥0` (real and non-negative)
/// - `|c0|² + |c1|² = 1` (normalization)
/// - If `c0 == 0`, then `c1 == Cpx::One {}` (canonical phase)
///
/// This ensures a normalized state vector with a unique, phase-fixed representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NormPair<T: FloatExt + Hash> {
    /// Real, non-negative amplitude of the |0⟩ basis state.
    pub c0: T,
    /// Complex amplitude of the |1⟩ basis state.
    pub c1: Cpx<T>,
}

/// A normalized single-qubit state in canonical form with orientation.
///
/// Wraps a `NormPair<T>` with a `BraKet` tag to indicate whether the state is a ket or bra.
/// The state is normalized and represented uniquely up to global phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NormQubit<T: FloatExt + Hash> {
    /// Indicates if the state is a ket or bra.
    pub bra_ket: BraKet,
    /// Canonical normalized amplitudes for the qubit state.
    pub norm_pair: NormPair<T>,
}

/// A mixed single-qubit state formed by a probabilistic mixture of a normalized state and its orthogonal complement.
///
/// The state represents the mixture:
/// `ρ = prob × |ψ⟩⟨ψ| + (1 - prob) × |ψ⊥⟩⟨ψ⊥|`
///
/// - `prob ∈ [0.5, 1.0]` is the probability weight on the primary state.
/// - `norm_pair` defines the normalized pure state |ψ⟩ in canonical form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MixPair<T: FloatExt + Hash> {
    /// Probability assigned to the pure state |ψ⟩ (must satisfy 0.5 ≤ prob ≤ 1.0).
    pub prob: T,
    /// Canonical normalized amplitudes defining the pure state |ψ⟩.
    pub norm_pair: NormPair<T>,
}

/// A mixed single-qubit state with bra/ket orientation.
///
/// Wraps a `MixPair<T>` along with a `BraKet` tag to indicate state orientation of the `MixPair<T>`.
/// Represents a probabilistic mixture of a normalized qubit and its orthogonal complement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MixQubit<T: FloatExt + Hash> {
    /// Indicates if the `MixPair<T>` is in a ket or bra representation.
    pub bra_ket: BraKet,
    /// The probabilistic mixture of a pure state and its orthogonal complement.
    pub mix_pair: MixPair<T>,
}

/// A separable multi-qubit product state defined over an index interval.
///
/// The state is interpreted as a tensor product of individual qubit states over a specified
/// inclusive `interval`. Each mapped entry provides a qubit's canonical normalized state.
/// Missing indices are filled in using the default `padding` value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IdxToNorm<T: FloatExt + Hash> {
    /// Inclusive interval of qubit indices this product state spans: [start, end].
    pub interval: (usize, usize),
    /// Default qubit state used for indices in the interval not explicitly mapped.
    pub padding: NormPair<T>,
    /// Mapping from selected qubit indices to the normalized single-qubit states.
    pub idx_to_norm: BTreeMap<usize, NormPair<T>>,
}

/// A separable multi-qubit product state with bra/ket orientation.
///
/// Encapsulates a product of single-qubit states using a canonical index-to-state mapping (`IdxToNorm<T>`),
/// along with a `BraKet` tag indicating whether the state is a ket or bra.
/// Assumes qubits are independent and ordered by index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SepQubits<T: FloatExt + Hash> {
    /// Indicates if the full state is a ket or bra.
    pub bra_ket: BraKet,
    /// Mapping from indices to single-qubit normalized states, with padding.
    pub idx_to_norm: IdxToNorm<T>,
}

/// A separable multi-qubit mixed state, where each qubit is independently in a mixed state.
///
/// Each qubit is represented by a `MixPair<T>`, describing a probabilistic mixture of a normalized
/// pure state and its orthogonal complement. The `idx_to_mix` map assigns such states to selected
/// qubit indices, while `padding` supplies a default value for indices not explicitly mapped.
///
/// Assumptions:
/// - No entanglement between qubits (fully separable state).
/// - Local mixing occurs independently at each qubit site.
/// - The state is defined over the inclusive `interval` of indices, with canonical ordering.
/// - The `bra_ket` field indicates whether this is a bra or ket state vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IdxToMixQubits<T: FloatExt + Hash> {
    /// Indicates if the overall `MixPair<T>`s are in a ket or bra representation.
    pub bra_ket: BraKet,
    /// Inclusive interval of qubit indices this product state spans: [start, end].
    pub interval: (usize, usize),
    /// Default mixed state used for indices in the interval not explicitly mapped.
    pub padding: MixPair<T>,
    /// Mapping from selected qubit indices to independent mixed single-qubit states.
    pub idx_to_mix: BTreeMap<usize, MixPair<T>>,
}

/// Enum representing a type of rank-1 operator: either a projector or a nilpotent.
///
/// - `Projector` corresponds to `|ψ⟩⟨ψ|`, a Hermitian, idempotent rank-1 operator.
/// - `Nilpotent` corresponds to either `|ψ⟩⟨ψ⊥|` or `|ψ⊥⟩⟨ψ|`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProNil {
    /// A Hermitian rank-1 projector: `|ψ⟩⟨ψ|`.
    Projector,
    /// A rank-1 nilpotent operator: `|ψ⟩⟨ψ⊥|` or `|ψ⊥⟩⟨ψ|`.
    Nilpotent,
}

/// A rank-1 operator (projector or nilpotent) constructed from a normalized qubit state.
///
/// Combines a `NormPair<T>` with an operator type and orientation tag. The `BraKet` field determines
/// the side (bra or ket) of the defining vector. For `Nilpotent`, the orthogonal complement is implied
/// on the opposite side of the rank-1 outer product.
///
/// Examples:
/// - `Projector` + `BraKet::Ket` with `|ψ⟩` → `|ψ⟩⟨ψ|`
/// - `Nilpotent` + `BraKet::Bra` with `⟨ψ|` → `|ψ⊥⟩⟨ψ|`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rk1PN<T: FloatExt + Hash> {
    /// Operator type: projector or nilpotent.
    pub pro_nil: ProNil,
    /// Indicates whether the vector is a bra or ket.
    pub bra_ket: BraKet,
    /// Canonical normalized qubit state defining the operator.
    pub norm_pair: NormPair<T>,
}

/// A general rank-1 matrix of the form`M = scalar × |ket⟩⟨bra|`.
///
/// This representation explicitly separates the ket (column) and bra (row) components of the matrix.
/// The underlying vectors are normalized and canonically phase-fixed via `NormPair<T>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rk1KB<T: FloatExt + Hash> {
    /// The ket vector `|ψ⟩` defining the column part of the outer product.
    pub ket: NormPair<T>,
    /// The bra vector `⟨ϕ|` defining the row part of the outer product.
    pub bra: NormPair<T>,
}

/// Enum representing a general rank-1 matrix operator.
///
/// This can be either a structured operator (`ProNil`) such as a projector or nilpotent,
/// or a fully general rank-1 matrix formed from arbitrary normalized vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rank1<T: FloatExt + Hash> {
    /// A structured rank-1 operator: projector or nilpotent.
    ProNil(Rk1PN<T>),
    /// A general rank-1 matrix: `|ket⟩⟨bra|` with independent canonical vectors.
    Other(Rk1KB<T>),
}

/// Struct representing a raw 2x2 complex matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawMat<T: FloatExt + Hash> {
    /// A raw 2x2 complex matrix.
    pub mat: [[Cpx<T>; 2]; 2],
}

/// Represents a general 2×2 complex matrix using the Pauli (complex-quaternion) basis.
///
/// The matrix is expressed as a linear combination of Pauli matrices:
/// `M = a₀·I + a₁·X + a₂·Y + a₃·Z`, where each `aᵢ` is a complex coefficient (`Cpx<T>`).
///
/// This representation is useful for compact encode general 2×2 complex operators using the Pauli basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CpxQuaternion<T: FloatExt + Hash> {
    /// Coefficients `[a₀, a₁, a₂, a₃]` corresponding to the Pauli basis `{I, X, Y, Z}`, in that order.
    pub coefficients: [Cpx<T>; 4],
}

/// Represents a 2×2 Hermitian matrix using the Pauli (complex-quaternion) basis.
///
/// The matrix is expressed as a real linear combination of Pauli matrices:
/// `M = a₀·I + a₁·X + a₂·Y + a₃·Z`, where each `aᵢ` is a real coefficient (`T`).
///
/// This form compactly encodes Hermitian 2×2 operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RealQuaternion<T: FloatExt + Hash> {
    /// Real coefficients `[a₀, a₁, a₂, a₃]` corresponding to the Pauli basis `{I, X, Y, Z}`, in that order.
    pub coefficients: [T; 4],
}

/// Represents a 2×2 complex matrix with unit determinant, expressed in the Pauli (complex-quaternion) basis.
///
/// The matrix is written as a linear combination of Pauli matrices:
/// `M = a₀·I + a₁·X + a₂·Y + a₃·Z`, where each `aᵢ` is a complex coefficient.
///
/// This struct stores only the non-identity coefficients `(a₁, a₂, a₃)`
/// as fields `x`, `y`, and `z`, corresponding to the Pauli matrices `X`, `Y`, and `Z`.
/// The identity coefficient `a₀` is computed implicitly to ensure `det(M) = 1`, using:
///
/// ```text
/// det(M) = a₀² − a₁² − a₂² − a₃² = 1
/// ⇒ a₀ = sqrt(1 + a₁² + a₂² + a₃²)
/// ```
///
/// This compact form is well suited for representing special unitary matrices (SU(2)),
/// rotations on the Bloch sphere, and Pauli-based gate decompositions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Det1CpxQuaternion<T: FloatExt + Hash> {
    /// Coefficient of the Pauli X component.
    pub x: Cpx<T>,

    /// Coefficient of the Pauli Y component.
    pub y: Cpx<T>,

    /// Coefficient of the Pauli Z component.
    pub z: Cpx<T>,
}

/// Represents a 2×2 Hermitian matrix with unit determinant, expressed in the Pauli (real-quaternion) basis.
///
/// As with the complex case, the matrix is expressed as:
/// `M = a₀·I + a₁·X + a₂·Y + a₃·Z`, but each `aᵢ` is real, and the identity component
/// `a₀` is computed implicitly to satisfy the determinant constraint:
///
/// ```text
/// det(M) = a₀² − a₁² − a₂² − a₃² = 1
/// ⇒ a₀ = sqrt(1 + a₁² + a₂² + a₃²)
/// ```
///
/// This form is useful for encoding Hermitian, unit-determinant operators, such as traceless generators
/// of real-valued quantum dynamics or normalized observables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Det1RealQuaternion<T: FloatExt + Hash> {
    /// Coefficient of the Pauli X component.
    pub x: T,

    /// Coefficient of the Pauli Y component.
    pub y: T,

    /// Coefficient of the Pauli Z component.
    pub z: T,
}

/// Represents a 2×2 complex matrix with unit determinant, expressed in the Pauli basis.
///
/// This enum distinguishes between:
/// - `Hermitian`: real-valued Pauli coefficients (used for Hermitian unit-determinant matrices),
/// - `Other`: complex-valued Pauli coefficients (for general SU(2) matrices).
///
/// Both variants implicitly encode the identity component `a₀` to satisfy `det(M) = 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Det1<T: FloatExt + Hash> {
    /// A Hermitian matrix with real-valued Pauli components.
    Hermitian(Det1RealQuaternion<T>),

    /// A general complex matrix with potentially non-Hermitian structure.
    Other(Det1CpxQuaternion<T>),
}

/// Represents a 2×2 complex matrix of rank 1 or 2, in a compact structured form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RankOneTwo<T: FloatExt + Hash> {
    /// A rank-1 matrix of the form`M = |ket⟩⟨bra|`.
    ///
    /// This variant encodes a matrix as the outer product of a normalized ket and bra.
    Rank1(Rank1<T>),

    /// A full-rank 2×2 complex matrix with unit determinant.
    ///
    /// The matrix is expressed in the Pauli (quaternion) basis, and may be Hermitian or non-Hermitian.
    Rank2(Det1<T>),
}

/// Represents a tensor product of local 2×2 complex operators acting on specific qubit indices.
///
/// This structure encodes a sparse tensor product of single-qubit operators. Each mapped entry
/// specifies a `RankOneTwo<T>` acting on a particular qubit index. The full operator is
/// constructed as an ordered tensor product over a specified interval of qubit indices. Indices
/// not explicitly mapped use the provided `padding` operator.
///
/// This is useful for compactly representing structured operators such as Kraus operators.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalOps<T: FloatExt + Hash> {
    /// Inclusive interval of qubit indices spanned by the tensor product: `[start, end]`.
    pub interval: (usize, usize),

    /// Default `RankOneTwo<T>` used for unmapped qubit indices within the interval.
    pub padding: RankOneTwo<T>,

    /// Mapping from selected qubit indices to non-default local operators.
    pub idx_to_mat: BTreeMap<usize, RankOneTwo<T>>,
}

/// Returns the inclusive interval `[min, max]` spanned by the keys of a BTreeMap.
/// Falls back to `default` if the map is empty.
pub fn key_interval_or<K: Copy + Ord>(map: &BTreeMap<K, impl Sized>, default: (K, K)) -> (K, K) {
    match (map.keys().next().copied(), map.keys().next_back().copied()) {
        (Some(min), Some(max)) => (min, max),
        _ => default,
    }
}

/// Swaps the values at keys `i` and `j` in a BTreeMap if either is present.
/// No effect if both keys are absent.
pub fn swap_btree_entries<K: Ord + Copy, V>(map: &mut BTreeMap<K, V>, i: K, j: K) {
    if i == j {
        return;
    }

    let i_val = map.remove(&i);
    let j_val = map.remove(&j);

    if let Some(val) = j_val {
        map.insert(i, val);
    }
    if let Some(val) = i_val {
        map.insert(j, val);
    }
}

impl BraKet {
    /// Returns the Hermitian conjugate (dagger) of the current `BraKet` variant.
    ///
    /// - `Bra` becomes `Ket`
    /// - `Ket` becomes `Bra`
    ///
    /// This operation corresponds to taking the adjoint of a vector:
    /// - `|ψ⟩† = ⟨ψ|`
    /// - `⟨ψ|† = |ψ⟩`
    pub fn dag(&self) -> Self {
        match self {
            Bra => Ket,
            Ket => Bra,
        }
    }
}

impl<T: FloatExt + Hash> NormPair<T> {
    /// Conjugates the norm_pair.
    pub fn conj(&self) -> Self {
        Self {
            c0: self.c0,
            c1: self.c1.conj(),
        }
    }
    /// Returns an orthogonal normalized qubit state.
    ///
    /// Given a qubit `|ψ⟩ = c₀|0⟩ + c₁|1⟩`, this returns an orthogonal state
    /// `|ψ⊥⟩ = c₁*|0⟩ − c₀*|1⟩`, normalized and up to a global phase.
    pub fn orthogonal(&self) -> Self {
        let (c1_mag, c1_phase) = self.c1.factor_out();
        let new_c0 = c1_mag.re(); // This radius from c1 is always non-negative.

        match new_c0 {
            val if val == T::zero() => Self {
                c0: T::zero(),
                c1: Cpx::ONE,
            },
            val if val == T::one() => Self {
                c0: T::one(),
                c1: Cpx::ZERO,
            },
            _ => Self {
                c0: new_c0,
                c1: -c1_phase.conj() * self.c0,
            },
        }
    }
    /// Factors out a global amplitude and phase to produce a canonical form.
    ///
    /// This ensures a unique and hashable representation by:
    /// - Forcing `c0 ∈ ℝ≥0` (non-negative real),
    /// - Normalizing the pair to unit ℓ₂-norm,
    /// - Extracting any global phase into a separate complex scalar.
    ///
    /// Returns:
    /// - `None` if both amplitudes are zero.
    /// - Otherwise, returns a triple `(canonical, sqrt_prob, phase)` such that: original = sqrt_prob × phase × canonical
    ///   , where:
    /// - `canonical` is a `NormPair` with unit norm and canonical phase,
    /// - `sqrt_prob ∈ ℝ≥0` is the amplitude magnitude,
    /// - `phase ∈ ℂ` is the global phase factor.
    pub fn try_factor_out(&self) -> Option<(Self, T, Cpx<T>)> {
        let zero = T::zero();
        let one = T::one();

        match (self.c0 == zero, matches!(self.c1, Cpx::Zero {})) {
            (true, true) => None,
            (true, false) => {
                let (sqrt_prob, ph) = self.c1.factor_out();
                Some((
                    Self {
                        c0: zero,
                        c1: Cpx::ONE,
                    },
                    sqrt_prob.re(),
                    ph,
                ))
            }
            (false, true) => {
                let c0_sign = if self.c0 > zero {
                    Cpx::ONE
                } else {
                    Cpx::NEG_ONE
                };
                Some((
                    Self {
                        c0: one,
                        c1: Cpx::ZERO,
                    },
                    self.c0.abs(),
                    c0_sign,
                ))
            }
            (false, false) => {
                // Normalize and extract global phase from c0
                let norm = self.c0.hypot(self.c1.rad());
                let c0_sign = if self.c0 > zero { one } else { -one };
                let c0_real = (self.c0 / norm).abs();
                let c1_rel = self.c1 / norm * c0_sign;

                Some((
                    Self {
                        c0: c0_real,
                        c1: c1_rel,
                    },
                    norm,
                    Cpx::Real { re: c0_sign },
                ))
            }
        }
    }
}

impl<T: FloatExt + Hash> NormQubit<T> {
    /// Returns the Hermitian conjugate (dagger) of the qubit representation,
    /// flipping `Bra` ↔ `Ket` and complex-conjugating the internal coefficients.
    pub fn dag(&self) -> Self {
        Self {
            norm_pair: self.norm_pair.conj(),
            bra_ket: self.bra_ket.dag(),
        }
    }
    /// Converts the `NormQubit` to its bra representation.
    /// If it is currently a ket, then return the Hermitian conjugate (dagger).
    pub fn to_bra(&self) -> Self {
        match self.bra_ket {
            Bra => *self,
            Ket => self.dag(),
        }
    }
    /// Converts the `NormQubit` to its ket representation.
    /// If it is currently a bra, then return the Hermitian conjugate (dagger).
    pub fn to_ket(&self) -> Self {
        match self.bra_ket {
            Ket => *self,
            Bra => self.dag(),
        }
    }
    /// Returns a normalized qubit orthogonal to `self`.
    pub fn orthogonal(&self) -> Self {
        Self {
            bra_ket: self.bra_ket,
            norm_pair: self.norm_pair.orthogonal(),
        }
    }
    /// Factors out a global amplitude and phase from the normalized qubit state.
    ///
    /// This provides a canonical decomposition of the form: original = sqrt_prob × phase × canonical, where:
    /// - `canonical` is a `NormQubit` with unit ℓ₂-norm and canonical phase,
    /// - `sqrt_prob ∈ ℝ≥0` is the size of the original state vector,
    /// - `phase ∈ ℂ` is the global complex phase.
    ///
    /// The result ensures:
    /// - `norm_pair.c0 ∈ ℝ≥0`
    /// - `canonical` is uniquely determined up to this constraint
    ///
    /// Returns:
    /// - `None` if both amplitudes are zero.
    /// - `Some(( canonical, sqrt_prob, phase))` otherwise.
    pub fn try_factor_out(&self) -> Option<(Self, T, Cpx<T>)> {
        self.norm_pair
            .try_factor_out()
            .map(|(norm_pair, sqrt_prob, phase)| {
                (
                    Self {
                        norm_pair,
                        bra_ket: self.bra_ket,
                    },
                    sqrt_prob,
                    phase,
                )
            })
    }
    /// Computes the inner product ⟨self|ket⟩ between two normalized qubits.
    ///
    /// `self` is interpreted as a bra, and `ket` as a ket.
    /// Automatically converts representations if necessary.
    pub fn inner(&self, ket: &Self) -> Cpx<T> {
        let bra_pair = self.to_bra().norm_pair;
        let ket_pair = ket.to_ket().norm_pair;
        (bra_pair.c1 * ket_pair.c1) + (bra_pair.c0 * ket_pair.c0)
    }
    /// Computes the outer product `|self⟩⟨other|` between two normalized qubit states,
    /// returning a canonical rank-1 matrix representation.
    ///
    /// This method interprets:
    /// - `self` as a ket vector (`|ψ⟩`)
    /// - `bra` as a bra vector (`⟨φ|`)
    ///
    /// It automatically converts the internal orientations (`BraKet`) of each input
    /// to ensure proper `ket–bra` alignment. The resulting operator is classified as:
    ///
    /// - `Projector` if `⟨bra| == ⟨self|` (i.e., `|self⟩⟨self|`);
    /// - `Nilpotent` if `⟨bra| == ⟨self⊥|` (i.e., `|self⟩⟨self⊥|`);
    /// - `Rank1::Other` for a general outer product `|ket⟩⟨bra|`.
    ///
    /// This ensures canonical detection and representation of projector and nilpotent
    /// operators when applicable.
    pub fn outer(&self, bra: &Self) -> Rank1<T> {
        let ket = self.to_ket().norm_pair;
        let bra = bra.to_bra().norm_pair;
        let conj_ket = ket.conj();

        if conj_ket == bra {
            Rank1::ProNil(Rk1PN {
                pro_nil: Projector,
                bra_ket: Ket,
                norm_pair: ket,
            })
        } else if conj_ket.orthogonal() == bra {
            Rank1::ProNil(Rk1PN {
                pro_nil: Nilpotent,
                bra_ket: Ket,
                norm_pair: ket,
            })
        } else {
            Rank1::Other(Rk1KB { ket, bra })
        }
    }
}

impl<T: FloatExt + Hash> MixPair<T> {
    /// Returns the complex conjugate of this mixed state.
    ///
    /// Conjugates the underlying normalized state while preserving the probability weight.
    pub fn conj(&self) -> Self {
        Self {
            prob: self.prob,
            norm_pair: self.norm_pair.conj(),
        }
    }

    /// Returns the orthogonal mixed state with the same probability.
    ///
    /// Constructs a new mixed state whose primary component is orthogonal to the original.
    pub fn orthogonal(&self) -> Self {
        Self {
            prob: self.prob,
            norm_pair: self.norm_pair.orthogonal(),
        }
    }

    /// Computes the purity of the mixed state: `Trace(ρ²) = p² + (1 - p)²`.
    ///
    /// Purity ranges from `0.5` (maximally mixed) to `1.0` (pure state).
    pub fn purity(&self) -> T {
        let p = self.prob;
        p.powi(2) + (T::one() - p).powi(2)
    }

    /// Computes the von Neumann entropy of the mixed state.
    ///
    /// Returns `−p ln(p) − (1 − p) ln(1 − p)` if `p ∈ (0, 1)`, and `0` if pure.
    /// Entropy is zero for pure states and maximal (ln 2) when `p = 0.5`.
    pub fn entropy_vn(&self) -> T {
        let zero = T::zero();
        let one = T::one();
        let p = self.prob;
        if (p != one) && (p != zero) {
            -p * p.ln() - (one - p) * (one - p).ln()
        } else {
            zero
        }
    }
}

impl<T: FloatExt + Hash> MixQubit<T> {
    /// Returns the Hermitian conjugate (dagger) of this mixed qubit state.
    ///
    /// Flips the `BraKet` orientation while preserving the mixed content.
    pub fn dag(&self) -> Self {
        Self {
            bra_ket: self.bra_ket.dag(),
            mix_pair: self.mix_pair.conj(),
        }
    }

    /// Returns the orthogonal mixed state with the same orientation and probability.
    ///
    /// The primary and orthogonal components are swapped, maintaining the same weighting.
    pub fn orthogonal(&self) -> Self {
        Self {
            bra_ket: self.bra_ket,
            mix_pair: self.mix_pair.orthogonal(),
        }
    }

    /// Returns the purity `Trace(ρ²)` of the underlying mixed state.
    pub fn purity(&self) -> T {
        self.mix_pair.purity()
    }

    /// Returns the von Neumann entropy of the underlying mixed state.
    pub fn entropy_vn(&self) -> T {
        self.mix_pair.entropy_vn()
    }
}

impl<T: FloatExt + Hash> IdxToNorm<T> {
    /// Returns the complex conjugate of this `IdxToNorm`.
    ///
    /// Applies conjugation to all `NormPair<T>` entries and the padding.
    pub fn conj(&self) -> Self {
        Self {
            interval: self.interval,
            padding: self.padding.conj(),
            idx_to_norm: self
                .idx_to_norm
                .iter()
                .map(|(k, v)| (*k, v.conj()))
                .collect(),
        }
    }

    /// Factors out the global amplitude and phase from each `NormPair`,
    /// returning a new `IdxToNorm` with canonicalized entries,
    /// and extracting the total amplitude and phase.
    ///
    /// Returns:
    /// - `None` if any `NormPair` fails to factor (e.g., is zero),
    /// - Otherwise, returns `(canonicalized_self, total_sqrt_prob, total_phase)`.
    pub fn try_factor_out(&self) -> Option<(Self, T, Cpx<T>)> {
        let mut idx_to_norm = BTreeMap::new();
        let mut acc_sqrt_prob = T::one();
        let mut acc_phase = Cpx::ONE;

        for (&k, v) in &self.idx_to_norm {
            let (canon, sqrt_prob, phase) = v.try_factor_out()?;
            idx_to_norm.insert(k, canon);
            acc_sqrt_prob = acc_sqrt_prob * sqrt_prob;
            acc_phase *= phase;
        }

        Some((
            Self {
                interval: self.interval,
                padding: self.padding,
                idx_to_norm,
            },
            acc_sqrt_prob,
            acc_phase,
        ))
    }

    /// Returns the list of qubit indices that have explicitly assigned normalized states.
    pub fn indices_keys(&self) -> Vec<usize> {
        self.idx_to_norm.keys().copied().collect()
    }

    /// Returns the inclusive interval of explicitly assigned qubit indices, if any.
    ///
    /// This is equivalent to `(min_index, max_index)` over the `idx_to_norm` map.
    /// Returns `(None, None)` if the map is empty.
    pub fn try_interval(&self) -> (Option<usize>, Option<usize>) {
        let min = self.idx_to_norm.keys().next().copied();
        let max = self.idx_to_norm.keys().next_back().copied();
        (min, max)
    }

    /// Returns a new `IdxToNorm` with its interval adjusted to match the minimum and maximum keys
    /// present in the `idx_to_norm` map.
    ///
    /// This method is useful for automatically syncing the `interval` field with the actual
    /// span of explicitly defined qubit indices.
    ///
    /// # Returns
    /// - `Some(Self)` with the updated interval if `idx_to_norm` is non-empty.
    /// - `None` if `idx_to_norm` is empty, as there are no keys to determine an interval.
    pub fn try_fit_interval(&self) -> Option<Self> {
        let (min, max) = self.try_interval();
        Some(Self {
            interval: (min?, max?),
            padding: self.padding,
            idx_to_norm: self.idx_to_norm.clone(),
        })
    }

    /// Returns a new `IdxToNorm` where all indices in the interval are explicitly filled.
    ///
    /// For any index within `self.interval` that does not appear in `idx_to_norm`, this method
    /// inserts an entry with the `padding` value. The result is a fully populated BTreeMap with
    /// no missing indices in the defined interval.
    pub fn extract_padding(&self) -> Self {
        let mut idx_to_norm = self.idx_to_norm.clone();
        let (start, end) = self.interval;

        for i in start..=end {
            idx_to_norm.entry(i).or_insert(self.padding);
        }

        Self {
            interval: self.interval,
            padding: self.padding,
            idx_to_norm,
        }
    }

    /// Returns a new `IdxToNorm` with the specified interval, leaving all other fields unchanged.
    ///
    /// This method updates the `interval` field while cloning the existing `padding`
    /// and `idx_to_norm` map.
    pub fn set_interval(&self, interval: (usize, usize)) -> Self {
        Self {
            interval,
            padding: self.padding,
            idx_to_norm: self.idx_to_norm.clone(),
        }
    }

    /// Returns a new `IdxToNorm` with the specified padding value,
    /// leaving the interval and index-to-norm map unchanged.
    pub fn set_padding(&self, padding: NormPair<T>) -> Self {
        Self {
            interval: self.interval,
            padding,
            idx_to_norm: self.idx_to_norm.clone(),
        }
    }
    /// Returns a new `IdxToNorm` with the specified indices removed from `idx_to_norm`.
    ///
    /// This method creates a copy of the current structure and removes any entries whose keys
    /// match the provided indices. The original `interval` and `padding` are preserved.
    ///
    /// # Type Parameters
    /// - `I`: An iterable collection of `usize` indices to remove.
    ///
    /// # Arguments
    /// - `indices`: Any iterable over `usize` values (e.g., a `Vec`, array, range, or `HashSet`).
    ///
    /// # Returns
    /// A new `IdxToNorm` with the specified indices removed from the map.
    ///
    /// # Examples
    /// ```rust
    /// //let pruned = original.discard(vec![1, 2, 4]);
    /// //let pruned = original.discard([3, 5, 7]);
    /// //let pruned = original.discard(0..=10);
    /// ```
    pub fn discard<I: IntoIterator<Item = usize>>(&self, indices: I) -> Self {
        let mut new_map = self.idx_to_norm.clone();
        for idx in indices {
            new_map.remove(&idx);
        }

        Self {
            interval: self.interval,
            padding: self.padding,
            idx_to_norm: new_map,
        }
    }
    /// Returns a new `IdxToNorm` with additional entries merged into `idx_to_norm`.
    ///
    /// For each `(index, norm_pair)` in the provided `idx_to_norm` map, if the index is not already present
    /// in the current map, it is inserted. Existing entries are left unchanged.
    ///
    /// The resulting interval is updated to span the full range of indices in the combined map, unless the map
    /// becomes empty, in which case the original interval is retained.
    ///
    /// # Arguments
    ///
    /// * `idx_to_norm` - A `BTreeMap<usize, NormPair<T>>` containing new entries to insert.
    ///
    /// # Returns
    ///
    /// A new `IdxToNorm` with the merged map and updated interval.
    pub fn extend(&self, idx_to_norm: BTreeMap<usize, NormPair<T>>) -> Self {
        let mut new_map = self.idx_to_norm.clone();
        for (k, v) in idx_to_norm {
            new_map.entry(k).or_insert(v);
        }

        let interval = key_interval_or(&new_map, self.interval);

        Self {
            interval,
            padding: self.padding,
            idx_to_norm: new_map,
        }
    }

    /// Returns a new `IdxToNorm` with the given entries forcibly updated.
    ///
    /// For each `(index, norm_pair)` in the provided map:
    /// - Any existing entry at that index is removed.
    /// - The new entry is inserted unconditionally.
    ///
    /// The updated `idx_to_norm` map reflects all changes, and the interval is recomputed to
    /// span the full range of keys present in the new map (if non-empty).
    ///
    /// # Arguments
    ///
    /// * `idx_to_norm` - A `BTreeMap<usize, NormPair<T>>` containing entries to overwrite.
    ///
    /// # Returns
    ///
    /// A new `IdxToNorm` with the forcibly updated entries and recalculated interval.
    pub fn force_update(&self, idx_to_norm: BTreeMap<usize, NormPair<T>>) -> Self {
        let mut new_map = self.idx_to_norm.clone();

        for (k, v) in idx_to_norm {
            new_map.insert(k, v); // `insert` already replaces the value if key exists
        }

        let interval = key_interval_or(&new_map, self.interval);

        Self {
            interval,
            padding: self.padding,
            idx_to_norm: new_map,
        }
    }
    /// Returns a new `IdxToNorm` with the values at two indices swapped.
    ///
    /// If both indices exist in `idx_to_norm`, their values are swapped.
    /// If only one exists, its value is moved to the other index.
    /// If neither exist, the mapping is unchanged.
    ///
    /// # Arguments
    /// * `i` - The first index.
    /// * `j` - The second index.
    pub fn swap(&self, i: usize, j: usize) -> Self {
        if i == j {
            return self.clone(); // No-op
        }

        let mut new_map = self.idx_to_norm.clone();
        swap_btree_entries(&mut new_map, i, j);

        let interval = key_interval_or(&new_map, self.interval);

        Self {
            interval,
            padding: self.padding,
            idx_to_norm: new_map,
        }
    }
}

impl<T: FloatExt + Hash> SepQubits<T> {
    /// Returns the Hermitian conjugate of the separable multi-qubit state.
    pub fn dag(&self) -> Self {
        Self {
            bra_ket: self.bra_ket.dag(),
            idx_to_norm: self.idx_to_norm.conj(),
        }
    }

    /// Converts the `SepQubit` to its bra representation.
    /// If it is currently a ket, then return the Hermitian conjugate (dagger).
    pub fn to_bra(&self) -> Self {
        match self.bra_ket {
            Bra => self.clone(),
            Ket => self.dag(),
        }
    }

    /// Converts the `SepQubit` to its ket representation.
    /// If it is currently a bra, then return the Hermitian conjugate (dagger).
    pub fn to_ket(&self) -> Self {
        match self.bra_ket {
            Ket => self.clone(),
            Bra => self.dag(),
        }
    }

    /// Factors out a global amplitude and phase from all `NormPair`s in the `SepQubit`.
    ///
    /// Returns:
    /// - `None` if any entry fails to factor (e.g., all-zero),
    /// - Otherwise, the canonicalized structure and overall factor.
    ///
    /// Satisfies: `original = sqrt_prob × phase × canonical`.
    pub fn try_factor_out(&self) -> Option<(Self, T, Cpx<T>)> {
        let (idx_to_norm, sqrt_prob, phase) = self.idx_to_norm.try_factor_out()?;
        Some((
            Self {
                bra_ket: self.bra_ket,
                idx_to_norm,
            },
            sqrt_prob,
            phase,
        ))
    }

    /// Computes the inner product ⟨self|ket⟩ between two `SepQubits<T>`, using the union of their intervals.
    ///
    /// This method interprets `self` as a bra and `ket` as a ket, aligning their index intervals and
    /// applying padding where necessary. Only indices within the specified interval are used;
    /// entries outside the interval (even if present in the underlying `BTreeMap`) are ignored.
    ///
    /// The inner product is computed as a product of local inner products at each index:
    /// ⟨ψ|ϕ⟩ = ∏ᵢ ⟨ψᵢ|ϕᵢ⟩.
    /// Returns `Cpx::ZERO` early if any local inner product is zero, and skips multiplying by `1` for efficiency.
    pub fn inner(&self, ket: &Self) -> Cpx<T> {
        let bra = self.to_bra().idx_to_norm;
        let ket = ket.to_ket().idx_to_norm;

        let (min, max) = {
            let (bra_min, bra_max) = bra.interval;
            let (ket_min, ket_max) = ket.interval;
            (min(bra_min, ket_min), max(bra_max, ket_max))
        };

        let bra = bra.set_interval((min, max)).extract_padding();
        let ket = ket.set_interval((min, max)).extract_padding();

        let mut acc = Cpx::ONE;
        for idx in min..max {
            let NormPair { c0: b0, c1: b1 } = bra.idx_to_norm[&idx];
            let NormPair { c0: k0, c1: k1 } = ket.idx_to_norm[&idx];
            let inner = b1 * k1 + b0 * k0;
            if matches!(inner, Cpx::Zero {}) {
                return Cpx::ZERO;
            } else if matches!(inner, Cpx::One {}) {
                continue;
            }
            acc *= inner;
        }

        acc
    }

    /// Returns a vector of all explicit qubit indices.
    pub fn indices_keys(&self) -> Vec<usize> {
        self.idx_to_norm.indices_keys()
    }

    /// Returns the inclusive interval of explicitly assigned qubit indices, if any.
    ///
    /// This is equivalent to `(min_index, max_index)` over the `idx_to_norm` map.
    /// Returns `(None, None)` if the map is empty.
    pub fn try_interval(&self) -> (Option<usize>, Option<usize>) {
        let min = self.idx_to_norm.idx_to_norm.keys().next().copied();
        let max = self.idx_to_norm.idx_to_norm.keys().next_back().copied();
        (min, max)
    }

    /// Attempts to shrink the interval to fit the range of explicitly defined indices.
    /// Returns `None` if `idx_to_norm` is empty.
    pub fn try_fit_interval(&self) -> Option<Self> {
        self.idx_to_norm.try_fit_interval().map(|idx_to_norm| Self {
            bra_ket: self.bra_ket,
            idx_to_norm,
        })
    }

    /// Returns a new `SepQubits` where all missing entries in the interval are filled with padding.
    pub fn extract_padding(&self) -> Self {
        Self {
            bra_ket: self.bra_ket,
            idx_to_norm: self.idx_to_norm.extract_padding(),
        }
    }

    /// Returns a new `SepQubits` with the interval manually updated.
    pub fn set_interval(&self, interval: (usize, usize)) -> Self {
        Self {
            bra_ket: self.bra_ket,
            idx_to_norm: self.idx_to_norm.set_interval(interval),
        }
    }

    /// Returns a new `SepQubits` with a new padding value.
    pub fn set_padding(&self, padding: NormPair<T>) -> Self {
        Self {
            bra_ket: self.bra_ket,
            idx_to_norm: self.idx_to_norm.set_padding(padding),
        }
    }

    /// Returns a new `SepQubits` with specified indices removed from the explicit assignments.
    pub fn discard<I: IntoIterator<Item = usize>>(&self, indices: I) -> Self {
        Self {
            bra_ket: self.bra_ket,
            idx_to_norm: self.idx_to_norm.discard(indices),
        }
    }

    /// Returns a new `SepQubits` with additional explicit assignments added if not already present.
    pub fn extend(&self, idx_to_norm: BTreeMap<usize, NormPair<T>>) -> Self {
        Self {
            bra_ket: self.bra_ket,
            idx_to_norm: self.idx_to_norm.extend(idx_to_norm),
        }
    }

    /// Returns a new `SepQubits` with the provided explicit assignments forcibly updated.
    pub fn force_update(&self, idx_to_norm: BTreeMap<usize, NormPair<T>>) -> Self {
        Self {
            bra_ket: self.bra_ket,
            idx_to_norm: self.idx_to_norm.force_update(idx_to_norm),
        }
    }

    /// Returns a new `SepQubits` with entries at indices `i` and `j` swapped.
    ///
    /// If only one of the two exists in the current map, its value is moved to the other index.
    /// If neither exists, the map is unchanged.
    pub fn swap(&self, i: usize, j: usize) -> Self {
        Self {
            bra_ket: self.bra_ket,
            idx_to_norm: self.idx_to_norm.swap(i, j),
        }
    }
}

impl<T: FloatExt + Hash> IdxToMixQubits<T> {
    /// Returns the Hermitian conjugate of the separable multi-qubit mixed state.
    pub fn dag(&self) -> Self {
        Self {
            bra_ket: self.bra_ket.dag(),
            interval: self.interval,
            padding: self.padding.conj(),
            idx_to_mix: self
                .idx_to_mix
                .iter()
                .map(|(k, v)| (*k, v.conj()))
                .collect(),
        }
    }
    /// Returns a vector of all explicit qubit indices.
    pub fn indices_keys(&self) -> Vec<usize> {
        self.idx_to_mix.keys().copied().collect()
    }

    /// Returns the inclusive interval of explicitly assigned qubit indices, if any.
    ///
    /// This is equivalent to `(min_index, max_index)` over the `idx_to_mix` map.
    /// Returns `(None, None)` if the map is empty.
    pub fn try_interval(&self) -> (Option<usize>, Option<usize>) {
        let min = self.idx_to_mix.keys().next().copied();
        let max = self.idx_to_mix.keys().next_back().copied();
        (min, max)
    }

    /// Attempts to shrink the interval to fit the range of explicitly defined indices.
    /// Returns `None` if `idx_to_mix` is empty.
    pub fn try_fit_interval(&self) -> Option<Self> {
        let (min, max) = self.try_interval();
        Some(Self {
            bra_ket: self.bra_ket,
            interval: (min?, max?),
            padding: self.padding,
            idx_to_mix: self.idx_to_mix.clone(),
        })
    }

    /// Returns a new `IdxToMixQubits` where all missing entries in the interval are filled with padding.
    pub fn extract_padding(&self) -> Self {
        let mut idx_to_mix = self.idx_to_mix.clone();
        let (start, end) = self.interval;

        for i in start..=end {
            idx_to_mix.entry(i).or_insert(self.padding);
        }

        Self {
            bra_ket: self.bra_ket,
            interval: self.interval,
            padding: self.padding,
            idx_to_mix,
        }
    }

    /// Returns a new `IdxToMixQubits` with the interval manually updated.
    pub fn set_interval(&self, interval: (usize, usize)) -> Self {
        Self {
            bra_ket: self.bra_ket,
            interval,
            padding: self.padding,
            idx_to_mix: self.idx_to_mix.clone(),
        }
    }

    /// Returns a new `IdxToMixQubits` with a new padding value.
    pub fn set_padding(&self, padding: MixPair<T>) -> Self {
        Self {
            bra_ket: self.bra_ket,
            interval: self.interval,
            padding,
            idx_to_mix: self.idx_to_mix.clone(),
        }
    }

    /// Returns a new `IdxToMixQubits` with specified indices removed from the explicit assignments.
    pub fn discard<I: IntoIterator<Item = usize>>(&self, indices: I) -> Self {
        let mut new_map = self.idx_to_mix.clone();
        for idx in indices {
            new_map.remove(&idx);
        }

        Self {
            bra_ket: self.bra_ket,
            interval: self.interval,
            padding: self.padding,
            idx_to_mix: new_map,
        }
    }

    /// Returns a new `IdxToMixQubits` with additional explicit assignments added if not already present.
    pub fn extend(&self, idx_to_mix: BTreeMap<usize, MixPair<T>>) -> Self {
        let mut new_map = self.idx_to_mix.clone();
        for (k, v) in idx_to_mix {
            new_map.entry(k).or_insert(v);
        }

        let interval = key_interval_or(&new_map, self.interval);

        Self {
            bra_ket: self.bra_ket,
            interval,
            padding: self.padding,
            idx_to_mix: new_map,
        }
    }

    /// Returns a new `IdxToMixQubits` with the provided explicit assignments forcibly updated.
    pub fn force_update(&self, idx_to_mix: BTreeMap<usize, MixPair<T>>) -> Self {
        let mut new_map = self.idx_to_mix.clone();

        for (k, v) in idx_to_mix {
            new_map.insert(k, v); // `insert` already replaces the value if key exists
        }

        let interval = key_interval_or(&new_map, self.interval);

        Self {
            bra_ket: self.bra_ket,
            interval,
            padding: self.padding,
            idx_to_mix: new_map,
        }
    }

    /// Returns a new `IdxToMixQubits` with entries at indices `i` and `j` swapped.
    ///
    /// If only one of the two exists in the current map, its value is moved to the other index.
    /// If neither exists, the map is unchanged.
    pub fn swap(&self, i: usize, j: usize) -> Self {
        if i == j {
            return self.clone(); // No-op
        }

        let mut new_map = self.idx_to_mix.clone();
        swap_btree_entries(&mut new_map, i, j);

        let interval = key_interval_or(&new_map, self.interval);

        Self {
            bra_ket: self.bra_ket,
            interval,
            padding: self.padding,
            idx_to_mix: new_map,
        }
    }
}

impl<T: FloatExt + Hash> Rk1PN<T> {
    /// Factors out the global amplitude (probability weight) from the rank-1 projector or nilpotent operator.
    ///
    /// The underlying `norm_qubit` is decomposed into a canonical form and the scalar amplitude.
    /// This method returns a pair `(canonical, prob)` where:
    /// - `canonical` is a `Rk1PN` with a normalized pure qubit state (unit norm and canonical phase),
    /// - `prob` is the squared magnitude (`sqrt_prob²`) representing the overall probability weight.
    ///
    /// Returns:
    /// - `None` if the underlying state is zero (no meaningful decomposition).
    /// - `Some(( canonical, prob))` otherwise.
    pub fn try_factor_out(self) -> Option<(Self, T)> {
        self.norm_pair
            .try_factor_out()
            .map(|(norm, sqrt_prob, ..)| {
                (
                    Self {
                        bra_ket: self.bra_ket,
                        norm_pair: norm,
                        pro_nil: self.pro_nil,
                    },
                    sqrt_prob.powi(2),
                )
            })
    }
    /// Returns the Hermitian conjugate (dagger) of the nilpotent matrix representation.
    ///
    /// If the operator is a nilpotent matrix, it returns a new one with an orthogonal qubit.
    /// If it is a projector, it returns itself unchanged.
    pub fn dag(self) -> Self {
        match self.pro_nil {
            Nilpotent => {
                let norm_pair = self.norm_pair.orthogonal();
                Self {
                    bra_ket: self.bra_ket,
                    norm_pair,
                    pro_nil: Nilpotent,
                }
            }
            Projector => self,
        }
    }
    /// Returns the trace of a rank-1 projector or nilpotent matrix as a `Cpx<T>`.
    pub fn trace(self) -> Cpx<T> {
        let NormPair { c0, c1 } = self.norm_pair;

        let dual = match self.pro_nil {
            Projector => self.norm_pair.conj(),
            Nilpotent => self.norm_pair.orthogonal().conj(),
        };

        Cpx::Real { re: c0 * dual.c0 } + c1 * dual.c1
    }
}

impl<T: FloatExt + Hash> Rk1KB<T> {
    /// Factors out the combined amplitude and global phase from a general rank-1 operator.
    ///
    /// Validates that `ket` is a |ψ⟩ and `bra` is a ⟨ϕ|, then factors out their
    /// respective global phases and magnitudes, returning:
    ///
    /// - `None` if either component is the zero state or has incorrect role labels.
    /// - `Some(( canonical, prob, phase))` where:
    ///   - `canonical` is a normalized version of the operator with unit-norm states,
    ///   - `prob` is the product of the sqrt-probabilities,
    ///   - `phase` is the product of their global phases.
    pub fn try_factor_out(self) -> Option<(Self, T, Cpx<T>)> {
        let ket_factored = self.ket.try_factor_out();
        let bra_factored = self.bra.try_factor_out();

        match (ket_factored, bra_factored) {
            (Some((ket_norm, sqrt_ket, ph_ket)), Some((bra_norm, sqrt_bra, ph_bra))) => {
                let sqrt_prob = sqrt_ket * sqrt_bra;
                let phase = ph_ket * ph_bra;

                Some((
                    Self {
                        ket: ket_norm,
                        bra: bra_norm,
                    },
                    sqrt_prob,
                    phase,
                ))
            }
            _ => None,
        }
    }
    /// Returns the Hermitian conjugate (dagger) of the `Rk1KB` matrix representation.
    ///
    /// This swaps the bra and ket components and applies conjugation to each.
    pub fn dag(self) -> Self {
        Self {
            ket: self.bra.conj(),
            bra: self.ket.conj(),
        }
    }
    /// Returns the trace of a general rank-1 matrix that is neither a projector nor nilpotent, as a `Cpx<T>`.
    pub fn trace(self) -> Cpx<T> {
        let NormPair { c0: k0, c1: k1 } = self.ket;
        let NormPair { c0: b0, c1: b1 } = self.bra;

        Cpx::Real { re: k0 * b0 } + k1 * b1
    }
}

impl<T: FloatExt + Hash> Rank1<T> {
    /// Factors out amplitude and optional global phase from a rank-1 operator.
    ///
    /// - Returns `None` if underlying state(s) are zero.
    /// - `Some(( canonical, prob, phase_opt))` where:
    ///   - `canonical` is normalized (projector or outer product),
    ///   - `prob` is the total scalar amplitude (squared norm),
    ///   - `phase_opt` is `Some(phase)` for `Other`, `None` for `ProNil`.
    pub fn try_factor_out(self) -> Option<(Self, T, Option<Cpx<T>>)> {
        match self {
            Self::ProNil(pn) => pn
                .try_factor_out()
                .map(|(canon, prob)| (Self::ProNil(canon), prob, None)),
            Self::Other(other) => other
                .try_factor_out()
                .map(|(canon, prob, phase)| (Self::Other(canon), prob, Some(phase))),
        }
    }
    /// Returns the Hermitian conjugate (dagger) of the rank-1 matrix.
    ///
    /// For `ProNil`, the projector is preserved and the nilpotent vector is conjugated if applicable.
    /// For `Other`, the ket and bra are swapped and conjugated.
    pub fn dag(self) -> Self {
        match self {
            Rank1::ProNil(pn) => Rank1::ProNil(pn.dag()),
            Rank1::Other(other) => Rank1::Other(other.dag()),
        }
    }
    /// Returns the trace of a general rank-1 matrix as a `Cpx<T>`.
    pub fn trace(self) -> Cpx<T> {
        match self {
            Rank1::ProNil(pn) => pn.trace(),
            Rank1::Other(other) => other.trace(),
        }
    }
}

impl<T: FloatExt + Hash> RawMat<T> {
    /// Constructs a raw 2×2 complex matrix from a rank-1 representation.
    ///
    /// Computes the outer product `|ket⟩⟨bra|`, with `ket` and `bra` derived from
    /// either a projector/nilpotent pairing or explicit components.
    pub fn from_rank1(rk1: Rank1<T>) -> Self {
        let build_outer = |k0: Cpx<T>, k1: Cpx<T>, b0: Cpx<T>, b1: Cpx<T>| RawMat {
            mat: [[k0 * b0, k0 * b1], [k1 * b0, k1 * b1]],
        };

        let (ket, bra) = match rk1 {
            Rank1::ProNil(pn) => {
                let dual = pn.norm_pair.conj();
                let dual_out = match pn.pro_nil {
                    Projector => dual,
                    Nilpotent => dual.orthogonal(),
                };
                match pn.bra_ket {
                    Ket => (pn.norm_pair, dual_out),
                    Bra => (dual_out, pn.norm_pair),
                }
            }
            Rank1::Other(other) => (other.ket, other.bra),
        };

        build_outer(
            Cpx::Real { re: ket.c0 },
            ket.c1,
            Cpx::Real { re: bra.c0 },
            bra.c1,
        )
    }

    /// Constructs a raw 2×2 matrix representing the density matrix of a mixed qubit state.
    ///
    /// The result is a convex combination of projectors onto a qubit and its orthogonal complement:
    /// ρ = `prob` × |ψ⟩⟨ψ| + (1 − `prob`) × |ψ⊥⟩⟨ψ⊥|.
    pub fn from_mixed_qubit(mixed: MixQubit<T>) -> Self {
        let prob = mixed.mix_pair.prob;
        let comp_prob = T::one() - prob;
        let norm = mixed.mix_pair.norm_pair;
        let bra_ket = mixed.bra_ket;

        let mk_proj = |norm_pair| {
            Self::from_rank1(Rank1::ProNil(Rk1PN {
                bra_ket,
                norm_pair,
                pro_nil: Projector,
            }))
        };

        let rho1 = mk_proj(norm) * prob;
        let rho2 = mk_proj(norm.orthogonal()) * comp_prob;

        rho1 + rho2
    }

    /// Converts a (complex) quaternion in the Pauli basis back into a raw 2×2 complex matrix.
    ///
    /// The complex quaternion is of the form:  
    /// `M = c₀·I + c₁·X + c₂·Y + c₃·Z`, where `coefficients = [c₀, c₁, c₂, c₃]`.
    pub fn from_cpx_quaternion(cq: CpxQuaternion<T>) -> Self {
        let [c0, c1, c2, c3] = cq.coefficients;
        let mat = [[c0 + c3, c1 - c2 * Cpx::J], [c1 + c2 * Cpx::J, c0 - c3]];
        Self { mat }
    }

    /// Converts a (real) quaternion in the Pauli basis back into a raw 2×2 complex matrix.
    ///
    /// The real quaternion is of the form:  
    /// `M = r₀·I + r₁·X + r₂·Y + r₃·Z`, where `coefficients = [r₀, r₁, r₂, r₃]`.
    pub fn from_real_quaternion(rq: RealQuaternion<T>) -> Self {
        Self::from_cpx_quaternion(CpxQuaternion::from_real_quaternion(rq))
    }

    /// Converts a unit-determinant complex quaternion into a raw 2×2 matrix.
    ///
    /// Computes the implicit identity coefficient `a₀` to satisfy `det(M) = 1`,
    /// then reconstructs the matrix from the full Pauli decomposition.
    pub fn from_det1_cpx_quaternion(d1cq: Det1CpxQuaternion<T>) -> Self {
        Self::from_cpx_quaternion(CpxQuaternion::from_det1_cpx_quaternion(d1cq))
    }

    /// Converts a unit-determinant real quaternion into a raw 2×2 matrix.
    ///
    /// Internally lifts the real coefficients into the complex domain and reconstructs
    /// the full matrix while enforcing the unit-determinant constraint.
    pub fn from_det1_real_quaternion(d1rq: Det1RealQuaternion<T>) -> Self {
        Self::from_real_quaternion(RealQuaternion::from_det1_real_quaternion(d1rq))
    }

    /// Returns the Hermitian conjugate (dagger) of the Raw matrix representation.
    pub fn dag(self) -> Self {
        let m = self.mat;
        Self {
            mat: [
                [m[0][0].conj(), m[1][0].conj()],
                [m[0][1].conj(), m[1][1].conj()],
            ],
        }
    }
    /// Returns the trace of a raw matrix as a `Cpx<T>`.
    pub fn trace(self) -> Cpx<T> {
        let m = self.mat;
        m[0][0] + m[1][1]
    }
    /// Returns the determinant of the matrix.
    pub fn det(self) -> Cpx<T> {
        let m = self.mat;
        (m[0][0] * m[1][1]) - (m[0][1] * m[1][0])
    }
    /// Returns `true` if all entries of the matrix are zero.
    pub fn is_zero(&self) -> bool {
        self.mat.iter().flatten().all(|c| matches!(c, Cpx::Zero {}))
    }
    /// Returns the rank of the matrix (0, 1, or 2).
    pub fn rank(&self) -> u8 {
        if self.is_zero() {
            0
        } else if matches!(self.det(), Cpx::Zero {}) {
            1
        } else {
            2
        }
    }
}

impl<T: FloatExt + Hash> CpxQuaternion<T> {
    /// Constructs a Pauli-basis (cpx-quaternion) representation from a raw 2×2 complex matrix.
    ///
    /// Decomposes the matrix `M` into the Pauli basis:  
    /// `M = a₀·I + a₁·X + a₂·Y + a₃·Z`,  
    /// where `a₀ = (M₀₀ + M₁₁)/2`, `a₁ = (M₀₁ + M₁₀)/2`, etc.
    pub fn from_raw_mat(raw: RawMat<T>) -> Self {
        let half = T::from(0.5).unwrap();
        let [[a, b], [c, d]] = raw.mat;

        let coefficients = [
            (a + d) * half,          // I component
            (b + c) * half,          // X component
            (b - c) * half * Cpx::J, // Y component
            (a - d) * half,          // Z component
        ];

        Self { coefficients }
    }

    /// Constructs a Pauli-basis (cpx-quaternion) representation from a Hermitian 2×2 matrix.
    pub fn from_real_quaternion(rq: RealQuaternion<T>) -> Self {
        let mut coefficients = [Cpx::ZERO; 4];
        for (i, &real) in rq.coefficients.iter().enumerate() {
            coefficients[i] = Cpx::Real { re: real };
        }
        Self { coefficients }
    }

    /// Constructs a `CpxQuaternion` from a `Det1CpxQuaternion`, computing the identity coefficient
    /// to satisfy `det(M) = 1` under the Pauli basis expansion.
    pub fn from_det1_cpx_quaternion(d1cq: Det1CpxQuaternion<T>) -> Self {
        let x2 = d1cq.x.powi(2);
        let y2 = d1cq.y.powi(2);
        let z2 = d1cq.z.powi(2);

        let a0 = (x2 + y2 + z2 + Cpx::ONE).powf(T::from(0.5).unwrap()); // a₀ = sqrt(1 + x² + y² + z²)

        let coefficients = [a0, d1cq.x, d1cq.y, d1cq.z];
        Self { coefficients }
    }

    /// Counts how many coefficients are not the `Cpx::Zero` variant.
    ///
    /// Assumes zeros are exactly represented by `Cpx::Zero {}`.
    pub fn nonzero_count(&self) -> usize {
        self.coefficients
            .iter()
            .filter(|c| !matches!(c, Cpx::Zero {}))
            .count()
    }

    /// Checks whether this matrix is an identity matrix or a Pauli matrix.
    pub fn is_ixyz(&self) -> bool {
        self.nonzero_count() == 1
    }

    /// Returns the Hermitian conjugate (dagger) of the cpx-quaternion representation.
    pub fn dag(&self) -> Self {
        let mut coefficients = self.coefficients;
        for elem in &mut coefficients {
            *elem = elem.conj();
        }
        Self { coefficients }
    }

    /// Returns the trace of the cpx-quaternion representation as a `Cpx<T>`.
    pub fn trace(&self) -> Cpx<T> {
        self.coefficients[0] * T::from(2.0).unwrap()
    }

    /// Computes the determinant of the matrix represented by this quaternion:
    /// det(M) = (a₀)² − (a₁)² − (a₂)² − (a₃)², where M = a₀·I + a₁·X + a₂·Y + a₃·Z.
    pub fn det(&self) -> Cpx<T> {
        let [a0, a1, a2, a3] = self.coefficients;
        a0.powi(2) - a1.powi(2) - a2.powi(2) - a3.powi(2)
    }
}

impl<T: FloatExt + Hash> RealQuaternion<T> {
    /// Constructs a `RealQuaternion` from a `Det1RealQuaternion`, computing the identity coefficient
    /// to satisfy `det(M) = 1` under the Pauli basis expansion.
    pub fn from_det1_real_quaternion(d1rq: Det1RealQuaternion<T>) -> Self {
        let x2 = d1rq.x.powi(2);
        let y2 = d1rq.y.powi(2);
        let z2 = d1rq.z.powi(2);

        let a0 = (x2 + y2 + z2 + T::one()).sqrt(); // a₀ = sqrt(1 + x² + y² + z²)

        let coefficients = [a0, d1rq.x, d1rq.y, d1rq.z];
        Self { coefficients }
    }

    /// Counts how many coefficients are not the `T::zero()` variant.
    ///
    /// Assumes zeros are exactly represented by `T::zero()`.
    pub fn nonzero_count(&self) -> usize {
        self.coefficients
            .iter()
            .filter(|c| **c != T::zero())
            .count()
    }
    /// Checks whether this matrix is an identity matrix or a Pauli matrix.
    pub fn is_ixyz(&self) -> bool {
        self.nonzero_count() == 1
    }
    /// Returns the trace of the real-quaternion representation as a `T`.
    pub fn trace(&self) -> T {
        self.coefficients[0] * T::from(2.0).unwrap()
    }

    /// Computes the determinant of the matrix represented by this real quaternion:
    /// det(M) = (a₀)² − (a₁)² − (a₂)² − (a₃)², where M = a₀·I + a₁·X + a₂·Y + a₃·Z.
    pub fn det(&self) -> T {
        let [a0, a1, a2, a3] = self.coefficients;
        a0.powi(2) - a1.powi(2) - a2.powi(2) - a3.powi(2)
    }
}

impl<T: FloatExt + Hash> Det1CpxQuaternion<T> {
    /// Constructs a `Det1CpxQuaternion` from a `Det1RealQuaternion`, lifting real components
    /// into the complex domain while preserving the determinant-1 constraint.
    pub fn from_real(d1rq: Det1RealQuaternion<T>) -> Self {
        Self {
            x: Cpx::Real { re: d1rq.x },
            y: Cpx::Real { re: d1rq.y },
            z: Cpx::Real { re: d1rq.z },
        }
    }

    /// Returns the implicit identity component `a₀` such that:
    /// `a₀ = sqrt(1 + x² + y² + z²)`, ensuring `det(M) = 1`.
    pub fn a0(&self) -> Cpx<T> {
        (Cpx::ONE + self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).powf(T::from(0.5).unwrap())
    }

    /// Returns the Hermitian conjugate (dagger) of the cpx-quaternion representation with unit determinant.
    pub fn dag(&self) -> Self {
        let x = self.x.conj();
        let y = self.y.conj();
        let z = self.z.conj();
        Self { x, y, z }
    }

    /// Returns the trace of the corresponding 2×2 matrix, which is `2·a₀`.
    pub fn trace(&self) -> Cpx<T> {
        self.a0() * T::from(2.0).unwrap()
    }

    /// Checks whether this matrix is an identity matrix or a Pauli matrix.
    pub fn is_ixyz(self) -> bool {
        CpxQuaternion::from_det1_cpx_quaternion(self).nonzero_count() == 1
    }
}

impl<T: FloatExt + Hash> Det1RealQuaternion<T> {
    /// Returns the implicit identity component `a₀` such that:
    /// `a₀ = sqrt(1 + x² + y² + z²)`, ensuring `det(M) = 1`.
    pub fn a0(&self) -> T {
        (T::one() + self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).powf(T::from(0.5).unwrap())
    }

    /// Returns the trace of the corresponding 2×2 matrix, which is `2·a₀`.
    pub fn trace(&self) -> T {
        self.a0() * T::from(2.0).unwrap()
    }

    /// Checks whether this matrix is an identity matrix or a Pauli matrix.
    pub fn is_ixyz(self) -> bool {
        RealQuaternion::from_det1_real_quaternion(self).nonzero_count() == 1
    }
}

impl<T: FloatExt + Hash> Det1<T> {
    /// Returns the trace of the matrix: `tr(M) = 2·a₀`, where `a₀` is the identity component.
    pub fn trace(&self) -> Cpx<T> {
        match self {
            Self::Hermitian(d1rq) => Cpx::Real { re: d1rq.trace() },
            Self::Other(d1cq) => d1cq.trace(),
        }
    }

    /// Returns the Hermitian adjoint (dagger) of the matrix.
    ///
    /// For Hermitian matrices, this is the identity operation.
    /// For non-Hermitian matrices, conjugates all Pauli coefficients.
    pub fn dag(&self) -> Self {
        match self {
            Self::Hermitian(d1rq) => Self::Hermitian(*d1rq),
            Self::Other(d1cq) => Self::Other(d1cq.dag()),
        }
    }

    /// Returns `true` if the matrix is exactly one of `{I, X, Y, Z}`.
    pub fn is_ixyz(self) -> bool {
        match self {
            Self::Hermitian(d1rq) => d1rq.is_ixyz(),
            Self::Other(d1cq) => d1cq.is_ixyz(),
        }
    }
}

impl<T: FloatExt + Hash> RankOneTwo<T> {
    /// Returns the Hermitian conjugate (dagger) of the matrix.
    pub fn dag(self) -> Self {
        match self {
            Self::Rank1(rk1) => Self::Rank1(rk1.dag()),
            Self::Rank2(det1) => Self::Rank2(det1.dag()),
        }
    }

    /// Returns the trace of the matrix.
    pub fn trace(self) -> Cpx<T> {
        match self {
            Self::Rank1(rk1) => rk1.trace(),
            Self::Rank2(det1) => det1.trace(),
        }
    }

    /// Returns the determinant of the matrix.
    pub fn det(self) -> Cpx<T> {
        match self {
            Self::Rank1(..) => Cpx::ZERO,
            Self::Rank2(..) => Cpx::ONE,
        }
    }

    /// Returns the rank of this finite rank 2x2 matrix.
    pub fn rank(self) -> usize {
        match self {
            Self::Rank1(..) => 1,
            Self::Rank2(..) => 2,
        }
    }
}

impl<T: FloatExt + Hash> LocalOps<T> {
    /// Returns the list of qubit indices that have explicitly assigned `RankOneTwo`.
    pub fn indices_keys(&self) -> Vec<usize> {
        self.idx_to_mat.keys().copied().collect()
    }

    /// Returns the inclusive interval of explicitly assigned qubit indices, if any.
    ///
    /// This is equivalent to `(min_index, max_index)` over the `idx_to_mix` map.
    /// Returns `(None, None)` if the map is empty.
    pub fn try_interval(&self) -> (Option<usize>, Option<usize>) {
        let min = self.idx_to_mat.keys().next().copied();
        let max = self.idx_to_mat.keys().next_back().copied();
        (min, max)
    }

    /// Returns a new `LocalOps` with its interval adjusted to match the minimum and maximum keys
    /// present in the `idx_to_mat` map.
    ///
    /// This method is useful for automatically syncing the `interval` field with the actual
    /// span of explicitly defined qubit indices.
    ///
    /// # Returns
    /// - `Some(Self)` with the updated interval if `idx_to_mat` is non-empty.
    /// - `None` if `idx_to_mat` is empty, as there are no keys to determine an interval.
    pub fn try_fit_interval(&self) -> Option<Self> {
        let (min, max) = self.try_interval();
        Some(Self {
            interval: (min?, max?),
            padding: self.padding,
            idx_to_mat: self.idx_to_mat.clone(),
        })
    }

    /// Returns a new `LocalOps` where all indices in the interval are explicitly filled.
    ///
    /// For any index within `self.interval` that does not appear in `idx_to_mat`, this method
    /// inserts an entry with the `padding` value. The result is a fully populated BTreeMap with
    /// no missing indices in the defined interval.
    pub fn extract_padding(&self) -> Self {
        let mut idx_to_mat = self.idx_to_mat.clone();
        let (start, end) = self.interval;

        for i in start..=end {
            idx_to_mat.entry(i).or_insert(self.padding);
        }

        Self {
            interval: self.interval,
            padding: self.padding,
            idx_to_mat,
        }
    }

    /// Returns a new `LocalOps` with the specified interval, leaving all other fields unchanged.
    ///
    /// This method updates the `interval` field while cloning the existing `padding`
    /// and `idx_to_mat` map.
    pub fn set_interval(&self, interval: (usize, usize)) -> Self {
        Self {
            interval,
            padding: self.padding,
            idx_to_mat: self.idx_to_mat.clone(),
        }
    }

    /// Returns a new `LocalOps` with the specified padding value,
    /// leaving the interval and index-to-mat map unchanged.
    pub fn set_padding(&self, padding: RankOneTwo<T>) -> Self {
        Self {
            interval: self.interval,
            padding,
            idx_to_mat: self.idx_to_mat.clone(),
        }
    }
    /// Returns a new `LocalOps` with the specified indices removed from `idx_to_mat`.
    ///
    /// This method creates a copy of the current structure and removes any entries whose keys
    /// match the provided indices. The original `interval` and `padding` are preserved.
    ///
    /// # Type Parameters
    /// - `I`: An iterable collection of `usize` indices to remove.
    ///
    /// # Arguments
    /// - `indices`: Any iterable over `usize` values (e.g., a `Vec`, array, range, or `HashSet`).
    ///
    /// # Returns
    /// A new `LocalOps` with the specified indices removed from the map.
    ///
    /// # Examples
    /// ```rust
    /// //let pruned = original.discard(vec![1, 2, 4]);
    /// //let pruned = original.discard([3, 5, 7]);
    /// //let pruned = original.discard(0..=10);
    /// ```
    pub fn discard<I: IntoIterator<Item = usize>>(&self, indices: I) -> Self {
        let mut new_map = self.idx_to_mat.clone();
        for idx in indices {
            new_map.remove(&idx);
        }

        Self {
            interval: self.interval,
            padding: self.padding,
            idx_to_mat: new_map,
        }
    }
    /// Returns a new `LocalOps` with additional entries merged into `idx_to_mat`.
    ///
    /// For each `(index, RankOneTwo<T>)` in the provided `idx_to_mat` map, if the index is not already present
    /// in the current map, it is inserted. Existing entries are left unchanged.
    ///
    /// The resulting interval is updated to span the full range of indices in the combined map, unless the map
    /// becomes empty, in which case the original interval is retained.
    ///
    /// # Arguments
    ///
    /// * `idx_to_mat` - A `BTreeMap<usize, RankOneTwo<T>>` containing new entries to insert.
    ///
    /// # Returns
    ///
    /// A new `LocalOps` with the merged map and updated interval.
    pub fn extend(&self, idx_to_mat: BTreeMap<usize, RankOneTwo<T>>) -> Self {
        let mut new_map = self.idx_to_mat.clone();
        for (k, v) in idx_to_mat.iter() {
            new_map.entry(*k).or_insert(*v);
        }

        let interval = key_interval_or(&new_map, self.interval);

        Self {
            interval,
            padding: self.padding,
            idx_to_mat: new_map,
        }
    }

    /// Returns a new `LocalOps` with the given entries forcibly updated.
    ///
    /// For each `(index, RankOneTwo<T>)` in the provided map:
    /// - Any existing entry at that index is removed.
    /// - The new entry is inserted unconditionally.
    ///
    /// The updated `idx_to_mat` map reflects all changes, and the interval is recomputed to
    /// span the full range of keys present in the new map (if non-empty).
    ///
    /// # Arguments
    ///
    /// * `idx_to_mat` - A `BTreeMap<usize, RankOneTwo<T>>` containing entries to overwrite.
    ///
    /// # Returns
    ///
    /// A new `LocalOps` with the forcibly updated entries and recalculated interval.
    pub fn force_update(&self, idx_to_mat: BTreeMap<usize, RankOneTwo<T>>) -> Self {
        let mut new_map = self.idx_to_mat.clone();

        for (k, v) in idx_to_mat {
            new_map.insert(k, v); // `insert` already replaces the value if key exists
        }

        let interval = key_interval_or(&new_map, self.interval);

        Self {
            interval,
            padding: self.padding,
            idx_to_mat: new_map,
        }
    }
    /// Returns a new `LocalOps` with the values at two indices swapped.
    ///
    /// If both indices exist in `idx_to_mat`, their values are swapped.
    /// If only one exists, its value is moved to the other index.
    /// If neither exist, the mapping is unchanged.
    ///
    /// # Arguments
    /// * `i` - The first index.
    /// * `j` - The second index.
    pub fn swap(&self, i: usize, j: usize) -> Self {
        if i == j {
            return self.clone(); // No-op
        }

        let mut new_map = self.idx_to_mat.clone();
        swap_btree_entries(&mut new_map, i, j);

        let interval = key_interval_or(&new_map, self.interval);

        Self {
            interval,
            padding: self.padding,
            idx_to_mat: new_map,
        }
    }
}

impl<T: FloatExt + Hash> Add for RawMat<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut mat = self.mat;
        for (i, row) in mat.iter_mut().enumerate() {
            for (j, elem) in row.iter_mut().enumerate() {
                *elem += other.mat[i][j];
            }
        }
        Self { mat }
    }
}

impl<T: FloatExt + Hash> AddAssign for RawMat<T> {
    fn add_assign(&mut self, other: Self) {
        for (i, row) in self.mat.iter_mut().enumerate() {
            for (j, elem) in row.iter_mut().enumerate() {
                *elem += other.mat[i][j];
            }
        }
    }
}

impl<T: FloatExt + Hash> Neg for RawMat<T> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut mat = self.mat;
        for row in mat.iter_mut() {
            for elem in row.iter_mut() {
                *elem = -*elem;
            }
        }
        Self { mat }
    }
}

impl<T: FloatExt + Hash> Sub for RawMat<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<T: FloatExt + Hash> SubAssign for RawMat<T> {
    fn sub_assign(&mut self, other: Self) {
        *self += -other;
    }
}

impl<T: FloatExt + Hash> Mul for RawMat<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let [[a, b], [c, d]] = self.mat;
        let [[e, f], [g, h]] = rhs.mat;

        Self {
            mat: [
                [a * e + b * g, a * f + b * h],
                [c * e + d * g, c * f + d * h],
            ],
        }
    }
}

impl<T: FloatExt + Hash> MulAssign for RawMat<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<T: FloatExt + Hash> Mul<T> for RawMat<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        let mut mat = self.mat;
        for row in mat.iter_mut() {
            for elem in row.iter_mut() {
                *elem *= rhs;
            }
        }
        Self { mat }
    }
}

impl<T: FloatExt + Hash> MulAssign<T> for RawMat<T> {
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T: FloatExt + Hash> Mul<Cpx<T>> for RawMat<T> {
    type Output = Self;
    fn mul(self, rhs: Cpx<T>) -> Self::Output {
        let mut mat = self.mat;
        for row in mat.iter_mut() {
            for elem in row.iter_mut() {
                *elem *= rhs;
            }
        }
        Self { mat }
    }
}

impl<T: FloatExt + Hash> MulAssign<Cpx<T>> for RawMat<T> {
    fn mul_assign(&mut self, rhs: Cpx<T>) {
        *self = *self * rhs;
    }
}

impl<T: FloatExt + Hash> Add for CpxQuaternion<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut coefficients = self.coefficients;
        for (i, elem) in coefficients.iter_mut().enumerate() {
            *elem += other.coefficients[i];
        }
        Self { coefficients }
    }
}

impl<T: FloatExt + Hash> AddAssign for CpxQuaternion<T> {
    fn add_assign(&mut self, other: Self) {
        let mut coefficients = self.coefficients;
        for (i, elem) in coefficients.iter_mut().enumerate() {
            *elem += other.coefficients[i];
        }
    }
}

impl<T: FloatExt + Hash> Neg for CpxQuaternion<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut coefficients = self.coefficients;
        for elem in coefficients.iter_mut() {
            *elem = -*elem;
        }
        Self { coefficients }
    }
}

impl<T: FloatExt + Hash> Sub for CpxQuaternion<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<T: FloatExt + Hash> SubAssign for CpxQuaternion<T> {
    fn sub_assign(&mut self, other: Self) {
        *self += -other;
    }
}

impl<T: FloatExt + Hash> Mul for CpxQuaternion<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mat_lhs = RawMat::from_cpx_quaternion(self);
        let mat_rhs = RawMat::from_cpx_quaternion(rhs);
        let mat_new = mat_lhs * mat_rhs;
        Self::from_raw_mat(mat_new)
    }
}

impl<T: FloatExt + Hash> MulAssign for CpxQuaternion<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: FloatExt + Hash> Mul<T> for CpxQuaternion<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        let mut coefficients = self.coefficients;
        for elem in coefficients.iter_mut() {
            *elem *= rhs;
        }
        Self { coefficients }
    }
}

impl<T: FloatExt + Hash> MulAssign<T> for CpxQuaternion<T> {
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T: FloatExt + Hash> Mul<Cpx<T>> for CpxQuaternion<T> {
    type Output = Self;
    fn mul(self, rhs: Cpx<T>) -> Self::Output {
        let mut coefficients = self.coefficients;
        for elem in coefficients.iter_mut() {
            *elem *= rhs;
        }
        Self { coefficients }
    }
}

impl<T: FloatExt + Hash> MulAssign<Cpx<T>> for CpxQuaternion<T> {
    fn mul_assign(&mut self, rhs: Cpx<T>) {
        *self = *self * rhs;
    }
}
