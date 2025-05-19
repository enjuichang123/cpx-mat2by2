//! The `cpx-mat2by2` library provides structures for representing and manipulating
//! quantum states and operators in a 2-dimensional Hilbert space (qubit).
//! It leverages the `cpx-coords` crate for complex number arithmetic.

use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use cpx_coords::*;

/// Represents the ket state \ket{0}.
pub const KET_Z0: State = State::Ket {
    c0: ONE,
    c1: ZERO,
    phase: ONE,
};
/// Represents the ket state \ket{1}.
pub const KET_Z1: State = State::Ket {
    c0: ZERO,
    c1: ONE,
    phase: ONE,
};
/// Represents the ket state \ket{+}.
pub const KET_X0: State = State::Ket {
    c0: INV_SQRT_2,
    c1: INV_SQRT_2,
    phase: ONE,
};
/// Represents the ket state \ket{-}.
pub const KET_X1: State = State::Ket {
    c0: INV_SQRT_2,
    c1: NEG_INV_SQRT_2,
    phase: ONE,
};
/// Represents the ket state \ket{+Y}.
pub const KET_Y0: State = State::Ket {
    c0: INV_SQRT_2,
    c1: J_INV_SQRT_2,
    phase: ONE,
};
/// Represents the ket state \ket{-Y}.
pub const KET_Y1: State = State::Ket {
    c0: INV_SQRT_2,
    c1: NEG_J_INV_SQRT_2,
    phase: ONE,
};

/// Enum representing the state of a qubit.
#[derive(Debug, Clone, Copy)]
pub enum State {
    /// Represents the Null state that should be avoided.
    Null,
    /// Represents a ket state.
    Ket { c0: Cpx, c1: Cpx, phase: Cpx },
    /// Represents a bra state.
    Bra { c0: Cpx, c1: Cpx, phase: Cpx },
}

/// Represents a (possibly weighted) projector operator derived from a normalized ket state.
#[derive(Debug, Clone, Copy)]
pub struct Projector {
    /// Represents the weight or scale of the projector.
    pub scalar: Cpx,
    /// Represents the projector by a given normalized ket state.
    pub ket: State,
}

/// Represents a (possibly weighted) nilpotent operator derived from a normalized ket state.
#[derive(Debug, Clone, Copy)]
pub struct Nilpotent {
    /// Represents the weight or scale of the nilpotent matrix.
    pub scalar: Cpx,
    /// Represents the nilpotent matrix by a given normalized ket state.
    pub ket: State,
}

/// A rank-1 matrix defined as M=scalar⋅∣ket⟩⟨bra∣M=scalar⋅∣ket⟩⟨bra∣, where the matrix is the product of a scalar, a ket vector, and a bra vector.
#[derive(Debug, Clone, Copy)]
pub struct Rank1 {
    /// Represents the weight or scale of the rank-1 matrix that is neither a projector nor a nilpotent matrix.
    pub scalar: Cpx,
    /// Represents the ket part of this rank-1 matrix.
    pub ket: State,
    /// Represents the bra part of this rank-1 matrix.
    pub bra: State,
}

/// Struct representing a rank-2 Pauli matrix.
#[derive(Debug, Clone, Copy)]
pub struct Pauli {
    /// idx values map to: 0 = I, 1 = X, 2 = Y, 3 = Z. Values outside this range are invalid.
    pub idx: u8,
    /// The scaling of this given Pauli matrix.
    pub scalar: Cpx,
}

/// Represents a full-rank 2×2 matrix using its singular value decomposition.
#[derive(Debug, Clone, Copy)]
pub struct Rank2SVD {
    /// First singular value (scaling factor for the first singular vector pair)
    pub sigma1: Cpx,
    /// Second singular value (scaling factor for the second singular vector pair)
    pub sigma2: Cpx,
    /// First left singular vector (ket)
    pub u1: State,
    /// Second left singular vector (ket)
    pub u2: State,
    /// First right singular vector (bra)
    pub v1: State,
    /// Second right singular vector (bra)
    pub v2: State,
}

/// Enum representing the classified complex 2-by-2 matrix.
#[derive(Debug, Clone, Copy)]
pub enum AltMat {
    /// Represents a rank-0 null matrix.
    Rank0,
    /// Represents a rank-1 projector matrix.
    Projector { projector: Projector },
    /// Represents a rank-1 nilpotent matrix.
    Nilpotent { nilpotent: Nilpotent },
    /// Represents a more general rank-1 matrix other than a projector or a nilpotent matrix.
    Rank1 { rank1: Rank1 },
    /// Represents a rank-2 Pauli matrix.
    Pauli { pauli: Pauli },
    /// Represents a more general rank-2 matrix other than a Pauli matrix.
    Rank2SVD { rank2: Rank2SVD },
}

impl State {
    /// Constructs a new Ket state. Returns `State::Null` if `phase == 0` or both `c0` and `c1` are zero.
    pub fn new_ket(c0: Cpx, c1: Cpx, phase: Cpx) -> Self {
        if phase.is_zero() || (c0.is_zero() && c1.is_zero()) {
            State::Null
        } else {
            State::Ket { c0, c1, phase }
        }
    }

    /// Constructs a new Bra state. Returns `State::Null` if `phase == 0` or both `c0` and `c1` are zero.
    pub fn new_bra(c0: Cpx, c1: Cpx, phase: Cpx) -> Self {
        if phase.is_zero() || (c0.is_zero() && c1.is_zero()) {
            State::Null
        } else {
            State::Bra { c0, c1, phase }
        }
    }

    /// Returns `true` if the state is a ket.
    pub fn is_ket(&self) -> bool {
        matches!(self, State::Ket { .. })
    }

    /// Returns `true` if the state is a bra.
    pub fn is_bra(&self) -> bool {
        matches!(self, State::Bra { .. })
    }

    /// Extracts the full complex global phase from the state.
    pub fn phase(&self) -> Cpx {
        match self {
            Self::Null => ZERO,
            Self::Bra { phase, c0, c1 } | Self::Ket { phase, c0, c1 } => {
                if phase.is_zero() {
                    ZERO
                } else if c0.is_zero() {
                    *phase * *c1
                } else if c1.is_zero() {
                    *phase * *c0
                } else {
                    let norm = c0.rad().hypot(c1.rad());
                    let rot = c0.rot();
                    *phase * norm * rot
                }
            }
        }
    }

    /// Normalizes a pair of complex coefficients with preserved relative phase.
    ///
    /// Returns `None` if both are zero. Otherwise, scales the pair to unit norm,
    /// sets global phase aside, and expresses relative phase via `PL`.
    ///
    /// - If one is zero, returns `(ZERO, ONE)` or `(ONE, ZERO)` accordingly.
    /// - Otherwise, returns `(Cpx::Real, Cpx::PL)` normalized and regularized.
    pub fn normalize_pair(c0: &Cpx, c1: &Cpx) -> Option<(Cpx, Cpx)> {
        if c0.is_zero() && c1.is_zero() {
            None
        } else if c0.is_zero() {
            Some((ZERO, ONE))
        } else if c1.is_zero() {
            Some((ONE, ZERO))
        } else {
            let r0 = c0.rad();
            let r1 = c1.rad();
            let norm = r0.hypot(r1);
            let phase_diff = c1.ph() - c0.ph();
            let new_c0 = Cpx::Real { re: r0 / norm };
            let new_c1 = Cpx::PL {
                rad: r1 / norm,
                ph: phase_diff,
            }
            .regularize();
            Some((new_c0, new_c1))
        }
    }

    /// Normalizes the state to a standard representation with `phase = ONE`.
    pub fn normalize(&self) -> Self {
        match self {
            Self::Null => Self::Null,
            Self::Ket { c0, c1, .. } => match Self::normalize_pair(c0, c1) {
                Some((nc0, nc1)) => Self::Ket {
                    c0: nc0,
                    c1: nc1,
                    phase: ONE,
                },
                None => Self::Null,
            },
            Self::Bra { c0, c1, .. } => match Self::normalize_pair(c0, c1) {
                Some((nc0, nc1)) => Self::Bra {
                    c0: nc0,
                    c1: nc1,
                    phase: ONE,
                },
                None => Self::Null,
            },
        }
    }

    /// Regularize the state to a standard representation.
    pub fn regularize(&self) -> Self {
        match self {
            Self::Null => Self::Null,
            Self::Ket { .. } | Self::Bra { .. } => self.phase().regularize() * self.normalize(),
        }
    }

    /// Get c0 from a state.
    pub fn c0(&self) -> Cpx {
        match self {
            Self::Null => ZERO,
            Self::Bra { phase, c0, c1 } | Self::Ket { phase, c0, c1 } => {
                if phase.is_zero() | c0.is_zero() {
                    ZERO
                } else if c1.is_zero() {
                    ONE
                } else {
                    let norm = c0.rad().hypot(c1.rad());
                    Cpx::Real {
                        re: c0.rad() / norm,
                    }
                }
            }
        }
    }
    /// Get c1 from a state.
    pub fn c1(&self) -> Cpx {
        match self {
            Self::Null => ZERO,
            Self::Bra { phase, c0, c1 } | Self::Ket { phase, c0, c1 } => {
                if phase.is_zero() | c1.is_zero() {
                    ZERO
                } else if c0.is_zero() {
                    ONE
                } else {
                    let norm = c0.rad().hypot(c1.rad());
                    *c1 / norm * c0.rot().conj()
                }
            }
        }
    }

    /// Apply a Hermitian conjugate (dagger) operation to convert between ket and bra.
    pub fn dag(&self) -> Self {
        match self {
            Self::Null => Self::Null,
            Self::Ket { c0, c1, phase } => Self::new_bra(c0.conj(), c1.conj(), phase.conj()),
            Self::Bra { c0, c1, phase } => Self::new_ket(c0.conj(), c1.conj(), phase.conj()),
        }
    }

    /// Returns the orthogonal state with normalized coefficients.
    ///
    /// Swaps and negates the coefficients to produce a perpendicular state.
    /// Returns `Null` if the state is `Null` or zero-valued.
    pub fn orthogonal(&self) -> Self {
        match self {
            Self::Null => Self::Null,
            Self::Ket { c0, c1, .. } => match Self::normalize_pair(c1, &-*c0) {
                Some((nc0, nc1)) => Self::new_ket(nc0, nc1, ONE),
                None => Self::Null,
            },
            Self::Bra { c0, c1, .. } => match Self::normalize_pair(c1, &-*c0) {
                Some((nc0, nc1)) => Self::new_bra(nc0, nc1, ONE),
                None => Self::Null,
            },
        }
    }

    /// Computes the inner product between a bra and a ket, returning a complex scalar.
    ///
    /// Accepts either (Bra, Ket) or (Ket, Bra). Returns `ZERO` if either state is `Null`.
    pub fn inner(&self, other: &State) -> Cpx {
        match (self, other) {
            (Self::Null, _) | (_, Self::Null) => ZERO,

            (
                Self::Bra { c0, c1, phase },
                Self::Ket {
                    c0: d0,
                    c1: d1,
                    phase: p2,
                },
            )
            | (
                Self::Ket { c0, c1, phase },
                Self::Bra {
                    c0: d0,
                    c1: d1,
                    phase: p2,
                },
            ) => (*phase * *p2 * (*c0 * *d0 + *c1 * *d1)).regularize(),

            _ => panic!("inner() is only defined between a Ket and a Bra."),
        }
    }

    /// Computes the outer product of a ket and a bra, yielding a rank-1 2×2 matrix.
    ///
    /// Accepts (Ket, Bra) or (Bra, Ket) order. Extracts phase and normalizes using `.regularize()`.
    pub fn outer(&self, other: &State) -> Option<Rank1> {
        let scalar = self.phase() * other.phase();
        let new_self = self.normalize();
        let new_other = other.normalize();

        match (new_self, new_other) {
            (Self::Ket { .. }, Self::Bra { .. }) | (Self::Bra { .. }, Self::Ket { .. }) => {
                let (ket, bra) = if matches!(new_self, Self::Ket { .. }) {
                    (new_self, new_other)
                } else {
                    (new_other, new_self)
                };

                Some(Rank1 { scalar, ket, bra })
            }
            _ => None,
        }
    }

    /// Constructs the projector |ψ⟩⟨ψ| or its orthogonal complement from a normalized state.
    ///
    /// If `is_orthogonal` is `true`, computes the projector for the orthogonal state.
    /// Returns `None` if the state is `Null`.
    pub fn projector(&self, is_orthogonal: bool) -> Option<Projector> {
        let state = if is_orthogonal {
            self.normalize().orthogonal()
        } else {
            self.normalize()
        };

        match state {
            Self::Null => None,
            Self::Ket { .. } => Some(Projector {
                scalar: ONE,
                ket: state,
            }),
            Self::Bra { .. } => Some(Projector {
                scalar: ONE,
                ket: state.dag(),
            }),
        }
    }
}
impl Projector {
    /// Regularize the projector to a standard representation.
    pub fn regularize(&self) -> Self {
        let mut new_ket = self.ket.regularize();
        let new_scalar = self.scalar * new_ket.phase();
        new_ket = new_ket.normalize();
        Projector {
            scalar: new_scalar,
            ket: new_ket,
        }
    }
    /// Normalize the projector to a standard representation.
    pub fn normalize(&self) -> Self {
        Projector {
            scalar: ONE,
            ket: self.ket.normalize(),
        }
    }
    /// Convert the projector into a [[Cpx; 2]; 2] form.
    pub fn to_mat(self) -> [[Cpx; 2]; 2] {
        let regularized = self.regularize();
        let r0 = regularized.ket.c0();
        let r1 = regularized.ket.c1();
        let s = regularized.scalar;
        let c0 = r0.conj();
        let c1 = r1.conj();
        [[s * r0 * c0, s * r0 * c1], [s * r1 * c0, s * r1 * c1]]
    }
    /// Take trace of this projector.
    pub fn tr(&self) -> Cpx {
        let regularized = self.regularize();
        let r0 = regularized.ket.c0();
        let r1 = regularized.ket.c1();
        let s = regularized.scalar;
        let c0 = r0.conj();
        let c1 = r1.conj();
        s * (r0 * c0 + r1 * c1)
    }
    /// Take dag of this projector.
    pub fn dag(&self) -> Self {
        Projector {
            scalar: self.scalar.conj(),
            ket: self.ket,
        }
    }
    /// Convert this projector into its complement.
    pub fn not(&self) -> Self {
        let regularized = self.regularize();
        Projector {
            scalar: regularized.scalar,
            ket: regularized.ket.orthogonal(),
        }
    }
    /// Check if this is indeed a null matrix.
    pub fn is_zero(&self) -> bool {
        self.regularize().scalar.is_zero()
    }
}
impl Nilpotent {
    /// Regularize the nilpotent matrix to a standard representation.
    pub fn regularize(&self) -> Self {
        let mut new_ket = self.ket.regularize();
        let new_scalar = self.scalar * new_ket.phase();
        new_ket = new_ket.normalize();
        Nilpotent {
            scalar: new_scalar,
            ket: new_ket,
        }
    }
    /// Normalize the nilpotent matrix to a standard representation.
    pub fn normalize(&self) -> Self {
        Nilpotent {
            scalar: ONE,
            ket: self.ket.normalize(),
        }
    }
    /// Convert the nilpotent matrix into a [[Cpx; 2]; 2] form.
    pub fn to_mat(self) -> [[Cpx; 2]; 2] {
        let regularized = self.regularize();
        let r0 = regularized.ket.c0();
        let r1 = regularized.ket.c1();
        let s = regularized.scalar;
        let c0 = r1.conj();
        let c1 = -r0.conj();
        [[s * r0 * c0, s * r0 * c1], [s * r1 * c0, s * r1 * c1]]
    }
    /// Take trace of this nilpotent matrix.
    pub fn tr(&self) -> Cpx {
        let regularized = self.regularize();
        let r0 = regularized.ket.c0();
        let r1 = regularized.ket.c1();
        let s = regularized.scalar;
        let c0 = r1.conj();
        let c1 = -r0.conj();
        s * (r0 * c0 + r1 * c1)
    }
    /// Take dag of this nilpotent matrix.
    pub fn dag(&self) -> Self {
        let regularized = self.regularize();
        let new_ket = State::Ket {
            c0: regularized.ket.c1(),
            c1: -regularized.ket.c0(),
            phase: regularized.ket.phase(),
        }
        .normalize();
        Nilpotent {
            scalar: regularized.scalar.conj(),
            ket: new_ket,
        }
    }
    /// Convert this nilpotent matrix into its complement.
    pub fn not(&self) -> Self {
        let regularized = self.regularize();
        Nilpotent {
            scalar: regularized.scalar,
            ket: regularized.ket.orthogonal(),
        }
    }
    /// Check if this matrix is indeed null.
    pub fn is_zero(&self) -> bool {
        self.regularize().scalar.is_zero()
    }
}
impl Rank1 {
    /// Regularizes the rank-1 matrix to a standard representation.
    pub fn regularize(&self) -> Self {
        let mut new_ket = self.ket.regularize();
        let mut new_bra = self.bra.regularize();
        let new_scalar = self.scalar * new_ket.phase() * new_bra.phase();
        new_ket = new_ket.normalize();
        new_bra = new_bra.normalize();
        Rank1 {
            scalar: new_scalar.regularize(),
            ket: new_ket,
            bra: new_bra,
        }
    }
    /// Normalizes the rank-1 matrix by setting the scalar to one.
    pub fn normalize(&self) -> Self {
        Rank1 {
            scalar: ONE,
            ket: self.ket.normalize(),
            bra: self.bra.normalize(),
        }
    }
    /// Converts this rank-1 operator into a full 2×2 matrix representation.
    pub fn to_mat(self) -> [[Cpx; 2]; 2] {
        let regularized = self.regularize();
        let r0 = regularized.ket.c0();
        let r1 = regularized.ket.c1();
        let c0 = regularized.bra.c0();
        let c1 = regularized.bra.c1();
        let s = regularized.scalar;
        [[s * r0 * c0, s * r0 * c1], [s * r1 * c0, s * r1 * c1]]
    }
    /// Returns the trace of the rank-1 matrix.
    pub fn tr(&self) -> Cpx {
        let regularized = self.regularize();
        let r0 = regularized.ket.c0();
        let r1 = regularized.ket.c1();
        let c0 = regularized.bra.c0();
        let c1 = regularized.bra.c1();
        let s = regularized.scalar;
        s * (r0 * c0 + r1 * c1)
    }
    /// Returns the Hermitian conjugate (dagger) of the rank-1 matrix.
    pub fn dag(&self) -> Self {
        let regularized = self.regularize();
        Rank1 {
            scalar: regularized.scalar.conj(),
            ket: regularized.bra.dag(),
            bra: regularized.ket.dag(),
        }
    }
    /// Returns the orthogonal complement of the rank-1 matrix.
    pub fn not(&self) -> Self {
        let regularized = self.regularize();
        Rank1 {
            scalar: regularized.scalar,
            ket: regularized.ket.orthogonal(),
            bra: regularized.bra.orthogonal(),
        }
    }
    /// Checks whether this rank-1 matrix is the zero operator.
    pub fn is_zero(&self) -> bool {
        self.regularize().scalar.is_zero()
    }
}
impl Pauli {
    /// Converts this Pauli matrix into its full 2×2 matrix form.
    pub fn to_mat(self) -> [[Cpx; 2]; 2] {
        match &self.idx {
            0 => [[self.scalar, ZERO], [ZERO, self.scalar]],
            1 => [[ZERO, self.scalar], [self.scalar, ZERO]],
            2 => [[ZERO, -J * self.scalar], [J * self.scalar, ZERO]],
            3 => [[self.scalar, ZERO], [ZERO, -self.scalar]],
            _ => panic!("Invalid input pauli index."),
        }
    }
}
impl Rank2SVD {
    /// Regularizes the rank-2 matrix to a canonical form by normalizing states, absorbing phases into singular values, and collapsing linearly dependent components.
    pub fn regularize(&self) -> Self {
        match (self.u1, self.u2, self.v1, self.v2) {
            (State::Ket { .. }, State::Ket { .. }, State::Bra { .. }, State::Bra { .. }) => {
                let mut new_u1 = self.u1.regularize();
                let mut new_u2 = self.u2.regularize();
                let mut new_v1 = self.v1.regularize();
                let mut new_v2 = self.v2.regularize();
                let mut sigma1 = self.sigma1 * new_u1.phase() * new_v1.phase();
                let sigma2 = self.sigma2 * new_u2.phase() * new_v2.phase();
                new_u1 = new_u1.normalize();
                new_u2 = new_u2.normalize();
                new_v1 = new_v1.normalize();
                new_v2 = new_v2.normalize();
                if new_u1 == new_u2 {
                    new_v1 = (new_v1 * sigma1 + new_v2 * sigma2).regularize();
                    sigma1 = new_v1.phase();
                    new_v1 = new_v1.normalize();
                    Rank2SVD {
                        sigma1,
                        sigma2: ZERO,
                        u1: new_u1,
                        u2: State::Null,
                        v1: new_v1,
                        v2: State::Null,
                    }
                } else if new_v1 == new_v2 {
                    new_u1 = (new_u1 * sigma1 + new_u2 * sigma2).regularize();
                    sigma1 = new_u1.phase();
                    new_u1 = new_u1.normalize();
                    Rank2SVD {
                        sigma1,
                        sigma2: ZERO,
                        u1: new_u1,
                        u2: State::Null,
                        v1: new_v1,
                        v2: State::Null,
                    }
                } else {
                    Rank2SVD {
                        sigma1,
                        sigma2,
                        u1: new_u1,
                        u2: new_u2,
                        v1: new_v1,
                        v2: new_v2,
                    }
                }
            }
            (State::Null, State::Null, State::Null, State::Null) => Rank2SVD {
                sigma1: ZERO,
                sigma2: ZERO,
                u1: State::Null,
                u2: State::Null,
                v1: State::Null,
                v2: State::Null,
            },
            _ => panic!("Invalid input."),
        }
    }
    /// Returns the determinant of the matrix.
    pub fn det(&self) -> Cpx {
        let s = self.regularize();
        s.sigma1 * s.sigma2
    }
    /// Converts this matrix into a 2×2 form: σ₁ u₁v₁† + σ₂ u₂v₂†.
    pub fn to_mat(self) -> [[Cpx; 2]; 2] {
        let m1 = Rank1 {
            scalar: self.sigma1,
            ket: self.u1,
            bra: self.v1,
        }
        .to_mat();
        let m2 = Rank1 {
            scalar: self.sigma2,
            ket: self.u2,
            bra: self.v2,
        }
        .to_mat();
        [
            [m1[0][0] + m2[0][0], m1[0][1] + m2[0][1]],
            [m1[1][0] + m2[1][0], m1[1][1] + m2[1][1]],
        ]
    }
    /// Returns the trace of the matrix.
    pub fn tr(&self) -> Cpx {
        let mat = &self.to_mat();
        mat[0][0] + mat[1][1]
    }
    /// Checks whether this matrix is the zero operator.
    pub fn is_zero(&self) -> bool {
        let s = self.regularize();
        s.sigma1.is_zero() && s.sigma2.is_zero()
    }
    /// Returns the rank of the matrix (0, 1, or 2).
    pub fn rank(&self) -> u8 {
        if self.is_zero() {
            0
        } else if self.det().is_zero() {
            1
        } else {
            2
        }
    }
    /// Extracts the coefficient of the identity component.
    pub fn get_i(&self) -> Cpx {
        let mat = &self.to_mat();
        (mat[0][0] + mat[1][1]) / 2.0
    }
    /// Extracts the coefficient of the Pauli-X component.
    pub fn get_x(&self) -> Cpx {
        let mat = &self.to_mat();
        (mat[0][1] + mat[1][0]) / 2.0
    }
    /// Extracts the coefficient of the Pauli-Y component.
    pub fn get_y(&self) -> Cpx {
        let mat = &self.to_mat();
        (mat[0][1] - mat[1][0]) / 2.0 * J
    }
    /// Extracts the coefficient of the Pauli-Z component.
    pub fn get_z(&self) -> Cpx {
        let mat = &self.to_mat();
        (mat[0][0] - mat[1][1]) / 2.0
    }
    /// Checks whether this matrix is a Pauli matrix.
    pub fn is_pauli(&self) -> bool {
        let b0 = !self.get_i().is_zero();
        let b1 = !self.get_x().is_zero();
        let b2 = !self.get_y().is_zero();
        let b3 = !self.get_z().is_zero();
        let count_nonzero = [b0, b1, b2, b3].iter().filter(|&&x| x).count(); // Count only `true` values
        count_nonzero == 1
    }
}
impl AltMat {
    /// Regularizes the classified complex 2×2 matrix to a standard representation.
    pub fn regularize(&self) -> Self {
        match self {
            Self::Rank0 => Self::Rank0,
            Self::Projector { projector } => Self::Projector {
                projector: projector.regularize(),
            },
            Self::Nilpotent { nilpotent } => Self::Nilpotent {
                nilpotent: nilpotent.regularize(),
            },
            Self::Rank1 { rank1 } => {
                if rank1.is_zero() {
                    Self::Rank0
                } else {
                    let regularized = rank1.regularize();
                    if regularized.ket == regularized.bra.dag() {
                        Self::Projector {
                            projector: Projector {
                                scalar: regularized.scalar,
                                ket: regularized.ket,
                            },
                        }
                    } else if regularized.bra.inner(&regularized.ket).is_zero() {
                        Self::Nilpotent {
                            nilpotent: Nilpotent {
                                scalar: regularized.scalar,
                                ket: regularized.ket,
                            },
                        }
                    } else {
                        Self::Rank1 { rank1: *rank1 }
                    }
                }
            }
            Self::Pauli { pauli } => Self::Pauli { pauli: *pauli },
            Self::Rank2SVD { rank2 } => {
                if rank2.is_zero() {
                    Self::Rank0
                } else if !rank2.det().is_zero() && !rank2.is_pauli() {
                    Self::Rank2SVD { rank2: *rank2 }.regularize()
                } else if !rank2.det().is_zero() && rank2.is_pauli() {
                    let c0 = rank2.get_i().regularize();
                    let c1 = rank2.get_x().regularize();
                    let c2 = rank2.get_y().regularize();
                    let c3 = rank2.get_z().regularize();
                    match (c0.is_zero(), c1.is_zero(), c2.is_zero()) {
                        (false, _, _) => Self::Pauli {
                            pauli: Pauli { idx: 0, scalar: c0 },
                        },
                        (true, false, _) => Self::Pauli {
                            pauli: Pauli { idx: 1, scalar: c1 },
                        },
                        (true, true, false) => Self::Pauli {
                            pauli: Pauli { idx: 2, scalar: c2 },
                        },
                        (true, true, true) => Self::Pauli {
                            pauli: Pauli { idx: 3, scalar: c3 },
                        },
                    }
                } else {
                    let mat = rank2.to_mat();
                    let b00: bool = mat[0][0].is_zero();
                    let b01: bool = mat[0][1].is_zero();
                    let b10: bool = mat[1][0].is_zero();
                    let b11: bool = mat[1][1].is_zero();

                    let bu0: bool = b00 && b01;
                    let bu1: bool = b10 && b11;
                    let bv0: bool = b00 && b10;
                    let bv1: bool = b01 && b11;

                    if bu0 {
                        let new_ket = State::Ket {
                            c0: ZERO,
                            c1: ONE,
                            phase: ONE,
                        };
                        let mut new_bra = State::Bra {
                            c0: mat[1][0],
                            c1: mat[1][1],
                            phase: ONE,
                        }
                        .regularize();
                        let new_scalar = new_bra.phase();
                        new_bra = new_bra.normalize();
                        if new_bra == new_ket.dag() {
                            Self::Projector {
                                projector: Projector {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                },
                            }
                        } else if new_bra.inner(&new_ket).is_zero() {
                            Self::Nilpotent {
                                nilpotent: Nilpotent {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                },
                            }
                        } else {
                            Self::Rank1 {
                                rank1: Rank1 {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                    bra: new_bra,
                                },
                            }
                        }
                    } else if bu1 {
                        let new_ket = State::Ket {
                            c0: ONE,
                            c1: ZERO,
                            phase: ONE,
                        };
                        let mut new_bra = State::Bra {
                            c0: mat[0][0],
                            c1: mat[0][1],
                            phase: ONE,
                        }
                        .regularize();
                        let new_scalar = new_bra.phase();
                        new_bra = new_bra.normalize();
                        if new_bra == new_ket.dag() {
                            Self::Projector {
                                projector: Projector {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                },
                            }
                        } else if new_bra.inner(&new_ket).is_zero() {
                            Self::Nilpotent {
                                nilpotent: Nilpotent {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                },
                            }
                        } else {
                            Self::Rank1 {
                                rank1: Rank1 {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                    bra: new_bra,
                                },
                            }
                        }
                    } else if bv0 {
                        let new_bra = State::Bra {
                            c0: ZERO,
                            c1: ONE,
                            phase: ONE,
                        };
                        let mut new_ket = State::Ket {
                            c0: mat[0][1],
                            c1: mat[1][1],
                            phase: ONE,
                        }
                        .regularize();
                        let new_scalar = new_ket.phase();
                        new_ket = new_ket.normalize();
                        if new_ket == new_bra.dag() {
                            Self::Projector {
                                projector: Projector {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                },
                            }
                        } else if new_bra.dag().inner(&new_ket).is_zero() {
                            Self::Nilpotent {
                                nilpotent: Nilpotent {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                },
                            }
                        } else {
                            Self::Rank1 {
                                rank1: Rank1 {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                    bra: new_bra,
                                },
                            }
                        }
                    } else if bv1 {
                        let new_bra = State::Bra {
                            c0: ONE,
                            c1: ZERO,
                            phase: ONE,
                        };
                        let mut new_ket = State::Ket {
                            c0: mat[0][0],
                            c1: mat[1][0],
                            phase: ONE,
                        }
                        .regularize();
                        let new_scalar = new_ket.phase();
                        new_ket = new_ket.normalize();
                        if new_ket == new_bra.dag() {
                            Self::Projector {
                                projector: Projector {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                },
                            }
                        } else if new_bra.dag().inner(&new_ket).is_zero() {
                            Self::Nilpotent {
                                nilpotent: Nilpotent {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                },
                            }
                        } else {
                            Self::Rank1 {
                                rank1: Rank1 {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                    bra: new_bra,
                                },
                            }
                        }
                    } else {
                        let frac_u1_u0 = mat[1][0] / mat[0][0];
                        let frac_v1_v0 = mat[0][1] / mat[0][0];
                        let mut new_ket = State::Ket {
                            c0: ONE,
                            c1: frac_u1_u0,
                            phase: ONE,
                        }
                        .regularize();
                        let mut new_bra = State::Bra {
                            c0: ONE,
                            c1: frac_v1_v0,
                            phase: ONE,
                        }
                        .regularize();
                        let new_scalar = mat[0][0] * new_ket.phase() * new_bra.phase();
                        new_ket = new_ket.normalize();
                        new_bra = new_bra.normalize();
                        if new_ket == new_bra.dag() {
                            Self::Projector {
                                projector: Projector {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                },
                            }
                        } else if new_bra.dag().inner(&new_ket).is_zero() {
                            Self::Nilpotent {
                                nilpotent: Nilpotent {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                },
                            }
                        } else {
                            Self::Rank1 {
                                rank1: Rank1 {
                                    scalar: new_scalar,
                                    ket: new_ket,
                                    bra: new_bra,
                                },
                            }
                        }
                    }
                }
            }
        }
    }
    /// Get the scalar of this matrix.
    pub fn scalar(&self) -> Cpx {
        match self {
            Self::Rank0 => ZERO,
            Self::Projector { projector } => projector.regularize().scalar,
            Self::Nilpotent { nilpotent } => nilpotent.regularize().scalar,
            Self::Rank1 { rank1 } => rank1.regularize().scalar,
            Self::Pauli { pauli } => pauli.scalar,
            Self::Rank2SVD { rank2 } => rank2.det().sqrt(),
        }
    }
    /// Normalize this matrix.
    pub fn normalize(&self) -> Self {
        match self.scalar().inv() {
            Ok(inv_scalar) => (*self * inv_scalar).regularize(),
            Err(_) => {
                eprintln!("Warning: Cannot normalize AltMat due to zero or invalid scalar.");
                *self // Return the original AltMat if inversion fails
            }
        }
    }
    /// Check is this matrix is indeed an identity matrix up to a scalar.
    pub fn is_id(&self) -> bool {
        let normalized = self.normalize();
        match normalized {
            Self::Pauli { pauli } => pauli.idx == 0,
            _ => false,
        }
    }
}

impl Hash for State {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let regularized = self.regularize();
        match regularized {
            State::Null => 0u8.hash(state),
            State::Ket { c0, c1, phase } => {
                1u8.hash(state);
                c0.hash(state);
                c1.hash(state);
                phase.hash(state);
            }
            State::Bra { c0, c1, phase } => {
                2u8.hash(state);
                c0.hash(state);
                c1.hash(state);
                phase.hash(state);
            }
        }
    }
}
impl Hash for Projector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let regularized = self.regularize();
        regularized.scalar.hash(state);
        regularized.ket.hash(state);
    }
}
impl Hash for Nilpotent {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let regularized = self.regularize();
        regularized.scalar.hash(state);
        regularized.ket.hash(state);
    }
}
impl Hash for Rank1 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let regularized = self.regularize();
        regularized.scalar.hash(state);
        regularized.ket.hash(state);
        regularized.bra.hash(state);
    }
}
impl Hash for Pauli {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.idx.hash(state);
        self.scalar.hash(state);
    }
}
impl Hash for Rank2SVD {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let regularized = self.regularize();
        regularized.sigma1.hash(state);
        regularized.sigma2.hash(state);
        regularized.u1.hash(state);
        regularized.u2.hash(state);
        regularized.v1.hash(state);
        regularized.v2.hash(state);
    }
}
impl Hash for AltMat {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let regularized = self.regularize();
        match regularized {
            AltMat::Rank0 => {
                0u8.hash(state);
            }
            AltMat::Projector { projector } => {
                1u8.hash(state);
                projector.hash(state);
            }
            AltMat::Nilpotent { nilpotent } => {
                2u8.hash(state);
                nilpotent.hash(state);
            }
            AltMat::Rank1 { rank1 } => {
                3u8.hash(state);
                rank1.hash(state);
            }
            AltMat::Pauli { pauli } => {
                4u8.hash(state);
                pauli.hash(state);
            }
            AltMat::Rank2SVD { rank2 } => {
                5u8.hash(state);
                rank2.hash(state);
            }
        }
    }
}

impl Neg for State {
    type Output = State;
    fn neg(self) -> State {
        let regularized = self.regularize();
        match regularized {
            Self::Null => Self::Null,
            Self::Ket { c0, c1, phase } => Self::Ket {
                c0,
                c1,
                phase: -phase,
            },
            Self::Bra { c0, c1, phase } => Self::Bra {
                c0,
                c1,
                phase: -phase,
            },
        }
    }
}
impl Neg for Projector {
    type Output = Projector;
    fn neg(self) -> Projector {
        Projector {
            scalar: -self.scalar,
            ket: self.ket,
        }
    }
}
impl Neg for Nilpotent {
    type Output = Nilpotent;
    fn neg(self) -> Nilpotent {
        Nilpotent {
            scalar: -self.scalar,
            ket: self.ket,
        }
    }
}
impl Neg for Rank1 {
    type Output = Rank1;
    fn neg(self) -> Self::Output {
        let regularized = self.regularize();
        Rank1 {
            scalar: -regularized.scalar,
            ket: regularized.ket,
            bra: regularized.bra,
        }
    }
}
impl Neg for Pauli {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Pauli {
            idx: self.idx,
            scalar: -self.scalar,
        }
    }
}
impl Neg for Rank2SVD {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let regularized = self.regularize();
        Rank2SVD {
            sigma1: -regularized.sigma1,
            sigma2: -regularized.sigma2,
            u1: regularized.u1,
            u2: regularized.u2,
            v1: regularized.v1,
            v2: regularized.v2,
        }
    }
}
impl Neg for AltMat {
    type Output = AltMat;
    fn neg(self) -> Self::Output {
        let regularized = self.regularize();
        match regularized {
            Self::Rank0 => Self::Rank0,
            Self::Projector { projector } => Self::Projector {
                projector: -projector,
            },
            Self::Nilpotent { nilpotent } => Self::Nilpotent {
                nilpotent: -nilpotent,
            },
            Self::Rank1 { rank1 } => Self::Rank1 { rank1: -rank1 },
            Self::Pauli { pauli } => Self::Pauli { pauli: -pauli },
            Self::Rank2SVD { rank2 } => Self::Rank2SVD { rank2: -rank2 },
        }
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        let s = self.regularize();
        let o = other.regularize();
        match (s, o) {
            (Self::Null, Self::Null) => true,
            (
                Self::Ket { c0, c1, phase },
                Self::Ket {
                    c0: d0,
                    c1: d1,
                    phase: p2,
                },
            ) => c0 == d0 && c1 == d1 && phase == p2,
            (
                Self::Bra { c0, c1, phase },
                Self::Bra {
                    c0: d0,
                    c1: d1,
                    phase: p2,
                },
            ) => c0 == d0 && c1 == d1 && phase == p2,
            _ => false,
        }
    }
}
impl Eq for State {}
impl PartialEq for Projector {
    fn eq(&self, other: &Self) -> bool {
        let s = self.regularize();
        let o = other.regularize();
        s.scalar == o.scalar && s.ket == o.ket
    }
}
impl Eq for Projector {}
impl PartialEq for Nilpotent {
    fn eq(&self, other: &Self) -> bool {
        let s = self.regularize();
        let o = other.regularize();
        s.scalar == o.scalar && s.ket == o.ket
    }
}
impl Eq for Nilpotent {}
impl PartialEq for Rank1 {
    fn eq(&self, other: &Self) -> bool {
        let s = self.regularize();
        let o = other.regularize();
        s.ket == o.ket && s.bra == o.bra && s.scalar == o.scalar
    }
}
impl Eq for Rank1 {}
impl PartialEq for Rank2SVD {
    fn eq(&self, other: &Self) -> bool {
        let s = self.regularize();
        let o = other.regularize();
        let c_same = s.sigma1 == o.sigma1
            && s.u1 == o.u1
            && s.v1 == o.v1
            && s.sigma2 == o.sigma2
            && s.u2 == o.u2
            && s.v2 == o.v2;
        let c_cross = s.sigma1 == o.sigma2
            && s.u1 == o.u2
            && s.v1 == o.v2
            && s.sigma2 == o.sigma1
            && s.u2 == o.u1
            && s.v2 == o.v1;
        c_same | c_cross
    }
}
impl Eq for Rank2SVD {}
impl PartialEq for Pauli {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.scalar == other.scalar
    }
}
impl Eq for Pauli {}
impl PartialEq for AltMat {
    fn eq(&self, other: &Self) -> bool {
        let s = self.regularize();
        let o = other.regularize();
        match (s, o) {
            (AltMat::Rank0, AltMat::Rank0) => true,
            (AltMat::Rank1 { rank1 }, AltMat::Rank1 { rank1: rk2 }) => rank1 == rk2,
            (AltMat::Rank2SVD { rank2 }, AltMat::Rank2SVD { rank2: rk2 }) => rank2 == rk2,
            (AltMat::Pauli { pauli }, AltMat::Pauli { pauli: p2 }) => pauli.idx == p2.idx,
            _ => false,
        }
    }
}
impl Eq for AltMat {}

impl Mul<State> for Cpx {
    type Output = State;
    fn mul(self, state: State) -> State {
        match state {
            State::Null => State::Null,
            State::Ket { c0, c1, phase } => {
                let new_phase = self * phase;
                if new_phase.is_zero() {
                    State::Null
                } else {
                    State::Ket {
                        c0,
                        c1,
                        phase: self * phase,
                    }
                }
            }
            State::Bra { c0, c1, phase } => {
                let new_phase = self * phase;
                if new_phase.is_zero() {
                    State::Null
                } else {
                    State::Bra {
                        c0,
                        c1,
                        phase: self * phase,
                    }
                }
            }
        }
    }
}
impl Mul<Cpx> for State {
    type Output = State;
    fn mul(self, scalar: Cpx) -> State {
        scalar * self
    }
}
impl Mul<State> for f32 {
    type Output = State;
    fn mul(self, state: State) -> State {
        match state {
            State::Null => State::Null,
            State::Ket { c0, c1, phase } => {
                let new_phase = self * phase;
                if new_phase.is_zero() {
                    State::Null
                } else {
                    State::Ket {
                        c0,
                        c1,
                        phase: self * phase,
                    }
                }
            }
            State::Bra { c0, c1, phase } => {
                let new_phase = self * phase;
                if new_phase.is_zero() {
                    State::Null
                } else {
                    State::Bra {
                        c0,
                        c1,
                        phase: self * phase,
                    }
                }
            }
        }
    }
}
impl Mul<f32> for State {
    type Output = State;
    fn mul(self, scalar: f32) -> State {
        scalar * self
    }
}
impl Mul<Cpx> for Projector {
    type Output = Projector;
    fn mul(self, rhs: Cpx) -> Projector {
        Projector {
            scalar: self.scalar * rhs,
            ket: self.ket,
        }
        .regularize()
    }
}
impl Mul<Projector> for Cpx {
    type Output = Projector;
    fn mul(self, rhs: Projector) -> Projector {
        (rhs * self).regularize()
    }
}
impl Mul<f32> for Projector {
    type Output = Projector;
    fn mul(self, rhs: f32) -> Projector {
        Projector {
            scalar: self.scalar * rhs,
            ket: self.ket,
        }
        .regularize()
    }
}
impl Mul<Projector> for f32 {
    type Output = Projector;
    fn mul(self, rhs: Projector) -> Projector {
        (rhs * self).regularize()
    }
}
impl Mul<Cpx> for Nilpotent {
    type Output = Nilpotent;
    fn mul(self, rhs: Cpx) -> Nilpotent {
        Nilpotent {
            scalar: self.scalar * rhs,
            ket: self.ket,
        }
        .regularize()
    }
}
impl Mul<Nilpotent> for Cpx {
    type Output = Nilpotent;
    fn mul(self, rhs: Nilpotent) -> Nilpotent {
        (rhs * self).regularize()
    }
}
impl Mul<f32> for Nilpotent {
    type Output = Nilpotent;
    fn mul(self, rhs: f32) -> Nilpotent {
        Nilpotent {
            scalar: self.scalar * rhs,
            ket: self.ket,
        }
        .regularize()
    }
}
impl Mul<Nilpotent> for f32 {
    type Output = Nilpotent;
    fn mul(self, rhs: Nilpotent) -> Nilpotent {
        (rhs * self).regularize()
    }
}
impl Mul<Cpx> for Rank1 {
    type Output = Rank1;
    fn mul(self, rhs: Cpx) -> Self::Output {
        Rank1 {
            scalar: self.scalar * rhs,
            ket: self.ket,
            bra: self.bra,
        }
        .regularize()
    }
}
impl Mul<Rank1> for Cpx {
    type Output = Rank1;
    fn mul(self, rhs: Rank1) -> Self::Output {
        Rank1 {
            scalar: rhs.scalar * self,
            ket: rhs.ket,
            bra: rhs.bra,
        }
        .regularize()
    }
}
impl Mul<f32> for Rank1 {
    type Output = Rank1;
    fn mul(self, rhs: f32) -> Self::Output {
        Rank1 {
            scalar: self.scalar * rhs,
            ket: self.ket,
            bra: self.bra,
        }
        .regularize()
    }
}
impl Mul<Rank1> for f32 {
    type Output = Rank1;
    fn mul(self, rhs: Rank1) -> Self::Output {
        Rank1 {
            scalar: rhs.scalar * self,
            ket: rhs.ket,
            bra: rhs.bra,
        }
        .regularize()
    }
}
impl Mul<Cpx> for Rank2SVD {
    type Output = Rank2SVD;
    fn mul(self, rhs: Cpx) -> Self::Output {
        let s = self.regularize();
        Rank2SVD {
            sigma1: s.sigma1 * rhs,
            sigma2: s.sigma2 * rhs,
            u1: s.u1,
            u2: s.u2,
            v1: s.v1,
            v2: s.v2,
        }
    }
}
impl Mul<Rank2SVD> for Cpx {
    type Output = Rank2SVD;
    fn mul(self, rhs: Rank2SVD) -> Self::Output {
        rhs * self
    }
}
impl Mul<f32> for Rank2SVD {
    type Output = Rank2SVD;
    fn mul(self, rhs: f32) -> Self::Output {
        let s = self.regularize();
        Rank2SVD {
            sigma1: s.sigma1 * rhs,
            sigma2: s.sigma2 * rhs,
            u1: s.u1,
            u2: s.u2,
            v1: s.v1,
            v2: s.v2,
        }
    }
}
impl Mul<Rank2SVD> for f32 {
    type Output = Rank2SVD;
    fn mul(self, rhs: Rank2SVD) -> Self::Output {
        rhs * self
    }
}
impl Mul<Cpx> for Pauli {
    type Output = Pauli;
    fn mul(self, rhs: Cpx) -> Self::Output {
        Pauli {
            idx: self.idx,
            scalar: self.scalar * rhs,
        }
    }
}
impl Mul<Pauli> for Cpx {
    type Output = Pauli;
    fn mul(self, rhs: Pauli) -> Self::Output {
        Pauli {
            idx: rhs.idx,
            scalar: rhs.scalar * self,
        }
    }
}
impl Mul<f32> for Pauli {
    type Output = Pauli;
    fn mul(self, rhs: f32) -> Self::Output {
        Pauli {
            idx: self.idx,
            scalar: self.scalar * rhs,
        }
    }
}
impl Mul<Pauli> for f32 {
    type Output = Pauli;
    fn mul(self, rhs: Pauli) -> Self::Output {
        Pauli {
            idx: rhs.idx,
            scalar: rhs.scalar * self,
        }
    }
}
impl Mul<Cpx> for AltMat {
    type Output = AltMat;
    fn mul(self, rhs: Cpx) -> Self::Output {
        match self {
            Self::Rank0 => Self::Rank0,
            Self::Projector { projector } => Self::Projector {
                projector: projector * rhs,
            },
            Self::Nilpotent { nilpotent } => Self::Nilpotent {
                nilpotent: nilpotent * rhs,
            },
            Self::Rank1 { rank1 } => Self::Rank1 { rank1: rank1 * rhs },
            Self::Pauli { pauli } => Self::Pauli { pauli: pauli * rhs },
            Self::Rank2SVD { rank2 } => Self::Rank2SVD { rank2: rank2 * rhs },
        }
    }
}
impl Mul<AltMat> for Cpx {
    type Output = AltMat;
    fn mul(self, rhs: AltMat) -> Self::Output {
        rhs * self
    }
}
impl Mul<f32> for AltMat {
    type Output = AltMat;
    fn mul(self, rhs: f32) -> Self::Output {
        match self {
            Self::Rank0 => Self::Rank0,
            Self::Projector { projector } => Self::Projector {
                projector: projector * rhs,
            },
            Self::Nilpotent { nilpotent } => Self::Nilpotent {
                nilpotent: nilpotent * rhs,
            },
            Self::Rank1 { rank1 } => Self::Rank1 { rank1: rank1 * rhs },
            Self::Pauli { pauli } => Self::Pauli { pauli: pauli * rhs },
            Self::Rank2SVD { rank2 } => Self::Rank2SVD { rank2: rank2 * rhs },
        }
    }
}
impl Mul<AltMat> for f32 {
    type Output = AltMat;
    fn mul(self, rhs: AltMat) -> Self::Output {
        rhs * self
    }
}

impl Add for State {
    type Output = State;
    fn add(self, other: State) -> State {
        match (self, other) {
            (Self::Null, _) => other,
            (_, Self::Null) => self,
            (
                Self::Ket { c0, c1, phase },
                Self::Ket {
                    c0: d0,
                    c1: d1,
                    phase: p2,
                },
            ) => {
                let new_c0 = phase * c0 + p2 * d0;
                let new_c1 = phase * c1 + p2 * d1;
                Self::Ket {
                    c0: new_c0,
                    c1: new_c1,
                    phase: ONE,
                }
                .regularize()
            }
            (
                Self::Bra { c0, c1, phase },
                Self::Bra {
                    c0: d0,
                    c1: d1,
                    phase: p2,
                },
            ) => {
                let new_c0 = phase * c0 + p2 * d0;
                let new_c1 = phase * c1 + p2 * d1;
                Self::Bra {
                    c0: new_c0,
                    c1: new_c1,
                    phase: ONE,
                }
                .regularize()
            }
            _ => panic!("Cannot add/sub State::Bra with State::Ket."),
        }
    }
}
impl AddAssign for State {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}
impl Sub for State {
    type Output = State;
    fn sub(self, other: State) -> State {
        self + (-other)
    }
}
impl SubAssign for State {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}
impl Add for AltMat {
    type Output = AltMat;

    fn add(self, rhs: Self) -> Self::Output {
        let s = self.regularize();
        let r = rhs.regularize();

        match (s, r) {
            (Self::Rank0, x) | (x, Self::Rank0) => x, // Return the non-zero rank matrix
            (x, y) => {
                // Extract matrices from each of the operands
                let m1 = match x {
                    AltMat::Rank1 { rank1 } => rank1.to_mat(),
                    AltMat::Rank2SVD { rank2 } => rank2.to_mat(),
                    AltMat::Pauli { pauli } => pauli.to_mat(),
                    _ => panic!("Invalid input."),
                };

                let m2 = match y {
                    AltMat::Rank1 { rank1 } => rank1.to_mat(),
                    AltMat::Rank2SVD { rank2 } => rank2.to_mat(),
                    AltMat::Pauli { pauli } => pauli.to_mat(),
                    _ => panic!("Invalid input."),
                };

                // Sum the two matrices element-wise
                let m3 = [
                    [m1[0][0] + m2[0][0], m1[0][1] + m2[0][1]],
                    [m1[1][0] + m2[1][0], m1[1][1] + m2[1][1]],
                ];

                // Return a Rank2SVD matrix with the summed result
                Self::Rank2SVD {
                    rank2: Rank2SVD {
                        sigma1: ONE,
                        sigma2: ONE,
                        u1: State::Ket {
                            c0: ONE,
                            c1: ZERO,
                            phase: ONE,
                        },
                        u2: State::Ket {
                            c0: ZERO,
                            c1: ONE,
                            phase: ONE,
                        },
                        v1: State::Bra {
                            c0: m3[0][0],
                            c1: m3[0][1],
                            phase: ONE,
                        },
                        v2: State::Bra {
                            c0: m3[1][0],
                            c1: m3[1][1],
                            phase: ONE,
                        },
                    }
                    .regularize(),
                }
                .regularize()
            }
        }
    }
}
impl AddAssign for AltMat {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}
impl Sub for AltMat {
    type Output = AltMat;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}
impl SubAssign for AltMat {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl Mul<State> for Projector {
    type Output = State;
    fn mul(self, rhs: State) -> Self::Output {
        match rhs {
            State::Null => State::Null,
            State::Ket { .. } => self.scalar * self.ket.dag().inner(&rhs) * self.ket,
            _ => panic!("Invalid input."),
        }
    }
}
impl Mul<Projector> for State {
    type Output = State;
    fn mul(self, rhs: Projector) -> Self::Output {
        match self {
            State::Null => State::Null,
            State::Bra { .. } => rhs.scalar * self.inner(&rhs.ket) * rhs.ket.dag(),
            _ => panic!("Invalid input."),
        }
    }
}
impl Mul<State> for Nilpotent {
    type Output = State;
    fn mul(self, rhs: State) -> Self::Output {
        match rhs {
            State::Null => State::Null,
            State::Ket { .. } => {
                let regularized = self.regularize();
                regularized.scalar
                    * regularized.ket.orthogonal().dag().inner(&rhs)
                    * regularized.ket
            }
            _ => panic!("Invalid input."),
        }
    }
}
impl Mul<Nilpotent> for State {
    type Output = State;
    fn mul(self, rhs: Nilpotent) -> Self::Output {
        match self {
            State::Null => State::Null,
            State::Bra { .. } => {
                let regularized = rhs.regularize();
                regularized.scalar
                    * self.inner(&regularized.ket)
                    * regularized.ket.orthogonal().dag()
            }
            _ => panic!("Invalid input."),
        }
    }
}
impl Mul<State> for Rank1 {
    type Output = State;
    fn mul(self, rhs: State) -> Self::Output {
        match rhs {
            State::Null => State::Null,
            State::Ket { .. } => {
                let new_rank1 = self.regularize();
                let new_rhs = rhs.regularize();
                let new_phase =
                    new_rank1.scalar * new_rhs.phase() * new_rank1.bra.inner(&new_rhs.normalize());
                let new_ket = new_rank1.ket;
                State::Ket {
                    c0: new_ket.c0(),
                    c1: new_ket.c1(),
                    phase: new_phase,
                }
                .regularize()
            }
            _ => panic!("Should have Rank1 * State::Ket instead of Rank1 * State::Bra."),
        }
    }
}
impl Mul<Rank1> for State {
    type Output = State;
    fn mul(self, rhs: Rank1) -> Self::Output {
        match self {
            State::Null => State::Null,
            State::Bra { .. } => {
                let new_rank1 = rhs.regularize();
                let new_self = self.regularize();
                let new_phase = new_rank1.scalar
                    * new_self.phase()
                    * new_self.inner(&new_rank1.ket.normalize());
                let new_bra = new_rank1.bra;
                State::Bra {
                    c0: new_bra.c0(),
                    c1: new_bra.c1(),
                    phase: new_phase,
                }
                .regularize()
            }
            _ => panic!("Should have State::Bra * Rank1 instead of State::Ket * Rank1."),
        }
    }
}
impl Mul<State> for Rank2SVD {
    type Output = State;
    fn mul(self, rhs: State) -> Self::Output {
        let s = rhs.regularize();
        match s {
            State::Null => State::Null,
            State::Ket { .. } => {
                let new_rank2svd = self.regularize();
                let rank1_1st = Rank1 {
                    scalar: new_rank2svd.sigma1,
                    ket: new_rank2svd.u1,
                    bra: new_rank2svd.v1,
                };
                let rank1_2nd = Rank1 {
                    scalar: new_rank2svd.sigma2,
                    ket: new_rank2svd.u2,
                    bra: new_rank2svd.v2,
                };
                (rank1_1st * s + rank1_2nd * s).regularize()
            }
            _ => panic!("Invalid input."),
        }
    }
}
impl Mul<Rank2SVD> for State {
    type Output = State;
    fn mul(self, rhs: Rank2SVD) -> Self::Output {
        let s = self.regularize();
        match s {
            State::Null => State::Null,
            State::Bra { .. } => {
                let new_rank2svd = rhs.regularize();
                let rank1_1st = Rank1 {
                    scalar: new_rank2svd.sigma1,
                    ket: new_rank2svd.u1,
                    bra: new_rank2svd.v1,
                };
                let rank1_2nd = Rank1 {
                    scalar: new_rank2svd.sigma2,
                    ket: new_rank2svd.u2,
                    bra: new_rank2svd.v2,
                };
                (s * rank1_1st + s * rank1_2nd).regularize()
            }
            _ => panic!("Invalid input."),
        }
    }
}
impl Mul<State> for Pauli {
    type Output = State;
    fn mul(self, rhs: State) -> Self::Output {
        match rhs {
            State::Null => State::Null,
            State::Ket { c0, c1, phase } => match self.idx {
                0 => self.scalar * rhs,
                1 => State::Ket {
                    c0: c1,
                    c1: c0,
                    phase: phase * self.scalar,
                }
                .regularize(),
                2 => State::Ket {
                    c0: -J * c1,
                    c1: J * c0,
                    phase: phase * self.scalar,
                }
                .regularize(),
                3 => State::Ket {
                    c0,
                    c1: -c1,
                    phase: phase * self.scalar,
                }
                .regularize(),
                _ => panic!("Invalid index for Pauli matrix."),
            },
            _ => panic!("Invalid input state. Please check when to use Ket or Bra."),
        }
    }
}

impl Mul<Pauli> for State {
    type Output = State;
    fn mul(self, rhs: Pauli) -> Self::Output {
        match self {
            State::Null => State::Null,
            State::Bra { c0, c1, phase } => match rhs.idx {
                0 => self * rhs.scalar,
                1 => State::Bra {
                    c0: c1,
                    c1: c0,
                    phase: phase * rhs.scalar,
                }
                .regularize(),
                2 => State::Bra {
                    c0: J * c1,
                    c1: -J * c0,
                    phase: phase * rhs.scalar,
                }
                .regularize(),
                3 => State::Bra {
                    c0,
                    c1: -c1,
                    phase: phase * rhs.scalar,
                }
                .regularize(),
                _ => panic!("Invalid index for Pauli matrix."),
            },
            _ => panic!("Invalid input state. Please check when to use Ket or Bra."),
        }
    }
}

impl Mul<State> for AltMat {
    type Output = State;
    fn mul(self, rhs: State) -> Self::Output {
        match rhs {
            State::Null => State::Null,
            State::Ket { .. } => match self {
                AltMat::Rank0 => State::Null,
                AltMat::Projector { projector } => projector * rhs,
                AltMat::Nilpotent { nilpotent } => nilpotent * rhs,
                AltMat::Rank1 { rank1 } => rank1 * rhs,
                AltMat::Rank2SVD { rank2 } => rank2 * rhs,
                AltMat::Pauli {
                    pauli: Pauli { idx, scalar },
                } => match idx {
                    0 => scalar * rhs,
                    1 => State::Ket {
                        c0: rhs.c1(),
                        c1: rhs.c0(),
                        phase: rhs.phase() * scalar,
                    }
                    .regularize(),
                    2 => State::Ket {
                        c0: -J * rhs.c1(),
                        c1: J * rhs.c0(),
                        phase: rhs.phase() * scalar,
                    }
                    .regularize(),
                    3 => State::Ket {
                        c0: rhs.c0(),
                        c1: -rhs.c1(),
                        phase: rhs.phase() * scalar,
                    },
                    _ => panic!("Invalid index for Pauli matrix."),
                },
            },
            State::Bra { .. } => {
                panic!("Should have AltMat * State::Ket instead of AltMat * State::Bra.")
            }
        }
    }
}

impl Mul<AltMat> for State {
    type Output = State;
    fn mul(self, rhs: AltMat) -> Self::Output {
        match self {
            State::Null => State::Null,
            State::Bra { .. } => match rhs {
                AltMat::Rank0 => State::Null,
                AltMat::Projector { projector } => self * projector,
                AltMat::Nilpotent { nilpotent } => self * nilpotent,
                AltMat::Rank1 { rank1 } => self * rank1,
                AltMat::Rank2SVD { rank2 } => self * rank2,
                AltMat::Pauli {
                    pauli: Pauli { idx, scalar },
                } => match idx {
                    0 => self * scalar,
                    1 => State::Ket {
                        c0: self.c1(),
                        c1: self.c0(),
                        phase: self.phase() * scalar,
                    },
                    2 => State::Ket {
                        c0: -J * self.c1(),
                        c1: J * self.c0(),
                        phase: self.phase() * scalar,
                    },
                    3 => State::Ket {
                        c0: self.c0(),
                        c1: -self.c1(),
                        phase: self.phase() * scalar,
                    },
                    _ => panic!("Invalid index for Pauli matrix."),
                },
            },
            State::Ket { .. } => {
                panic!("Should have State::Bra * AltMat instead of State::Ket * AltMat.")
            }
        }
    }
}
impl Mul for Rank1 {
    type Output = Rank1;
    fn mul(self, other: Self) -> Self::Output {
        let s = self.regularize();
        let o = other.regularize();
        let new_scalar = (s.scalar * o.scalar * s.bra.inner(&o.ket)).regularize();
        Rank1 {
            scalar: new_scalar,
            ket: s.ket,
            bra: o.bra,
        }
        .regularize()
    }
}
impl Mul for Rank2SVD {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let s = self.regularize();
        let r = rhs.regularize();
        Rank2SVD {
            sigma1: ONE,
            sigma2: ONE,
            u1: s.u1,
            u2: s.u2,
            v1: s.sigma1
                * (s.v1.inner(&r.u1) * r.v1 * r.sigma1 + s.v1.inner(&r.u2) * r.v2 * r.sigma2),
            v2: s.sigma2
                * (s.v2.inner(&r.u1) * r.v1 * r.sigma1 + s.v2.inner(&r.u2) * r.v2 * r.sigma2),
        }
        .regularize()
    }
}
impl Mul for Pauli {
    type Output = Pauli;
    fn mul(self, rhs: Self) -> Self::Output {
        let pauli_lookup = [
            [0, 1, 2, 3], // I * {I, X, Y, Z}
            [1, 0, 3, 2], // X * {I, X, Y, Z}
            [2, 3, 0, 1], // Y * {I, X, Y, Z}
            [3, 2, 1, 0], // Z * {I, X, Y, Z}
        ];
        let sign_lookup = [
            [ONE; 4],          // I * {I, X, Y, Z}
            [ONE, ONE, J, -J], // X * {I, X, Y, Z}
            [ONE, -J, ONE, J], // Y * {I, X, Y, Z}
            [ONE, J, -J, ONE], // Z * {I, X, Y, Z}
        ];
        let new_idx = pauli_lookup[self.idx as usize][rhs.idx as usize];
        let new_scalar =
            self.scalar * rhs.scalar * sign_lookup[self.idx as usize][rhs.idx as usize];
        Pauli {
            idx: new_idx,
            scalar: new_scalar,
        }
    }
}
impl Mul for AltMat {
    type Output = AltMat;
    fn mul(self, rhs: Self) -> Self::Output {
        match (self.regularize(), rhs.regularize()) {
            (Self::Rank0, _) | (_, Self::Rank0) => Self::Rank0,
            (
                Self::Projector { projector },
                Self::Projector {
                    projector: projector2,
                },
            ) => {
                if projector.ket == projector2.ket {
                    Self::Projector { projector }
                } else if projector.ket.dag().inner(&projector2.ket).is_zero() {
                    Self::Rank0
                } else {
                    Self::Rank1 {
                        rank1: Rank1 {
                            scalar: projector.scalar
                                * projector2.scalar
                                * projector.ket.dag().inner(&projector2.ket),
                            ket: projector.ket,
                            bra: projector2.ket.dag(),
                        },
                    }
                }
            }
            (Self::Projector { projector }, Self::Nilpotent { nilpotent }) => {
                if projector.ket == nilpotent.ket {
                    Self::Rank0
                } else if projector.ket.dag().inner(&nilpotent.ket).is_zero() {
                    Self::Nilpotent {
                        nilpotent: Nilpotent {
                            scalar: projector.scalar * nilpotent.scalar,
                            ket: nilpotent.ket,
                        },
                    }
                } else {
                    Self::Rank1 {
                        rank1: Rank1 {
                            scalar: projector.scalar
                                * nilpotent.scalar
                                * projector.ket.dag().inner(&nilpotent.ket),
                            ket: projector.ket,
                            bra: nilpotent.ket.dag(),
                        },
                    }
                }
            }
            (Self::Projector { projector }, Self::Rank1 { rank1 }) => Self::Rank1 {
                rank1: Rank1 {
                    scalar: projector.scalar * rank1.scalar * projector.ket.dag().inner(&rank1.ket),
                    ket: projector.ket,
                    bra: rank1.bra,
                },
            },
            (Self::Projector { projector }, Self::Rank2SVD { rank2 }) => {
                let mut new_bra = projector.scalar
                    * (rank2.sigma1 * rank2.v1 * projector.ket.dag().inner(&rank2.u1)
                        + rank2.sigma2 * rank2.v2 * projector.ket.dag().inner(&rank2.u2))
                    .regularize();
                let new_scalar = new_bra.phase();
                new_bra = new_bra.normalize();
                if projector.ket == new_bra.dag() {
                    Self::Projector {
                        projector: Projector {
                            scalar: new_scalar,
                            ket: projector.ket,
                        },
                    }
                } else if new_bra.inner(&projector.ket).is_zero() {
                    Self::Nilpotent {
                        nilpotent: Nilpotent {
                            scalar: new_scalar,
                            ket: projector.ket,
                        },
                    }
                } else {
                    Self::Rank1 {
                        rank1: Rank1 {
                            scalar: new_scalar,
                            ket: projector.ket,
                            bra: new_bra,
                        },
                    }
                }
            }
            (Self::Projector { projector }, Self::Pauli { pauli }) => match pauli.idx {
                0 => Self::Projector {
                    projector: projector * pauli.scalar,
                },
                1 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: projector.scalar * pauli.scalar,
                        ket: projector.ket,
                        bra: projector.ket.dag().inner(&KET_X0) * KET_X0.dag()
                            - projector.ket.dag().inner(&KET_X1) * KET_X1.dag(),
                    },
                }
                .regularize(),
                2 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: projector.scalar * pauli.scalar,
                        ket: projector.ket,
                        bra: projector.ket.dag().inner(&KET_Y0) * KET_Y0.dag()
                            - projector.ket.dag().inner(&KET_Y1) * KET_Y1.dag(),
                    },
                }
                .regularize(),
                3 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: projector.scalar * pauli.scalar,
                        ket: projector.ket,
                        bra: projector.ket.dag().inner(&KET_Z0) * KET_Z0.dag()
                            - projector.ket.dag().inner(&KET_Z1) * KET_Z1.dag(),
                    },
                }
                .regularize(),
                _ => unreachable!(),
            },
            (Self::Nilpotent { nilpotent }, Self::Projector { projector }) => {
                if projector.ket == nilpotent.ket {
                    Self::Rank0
                } else if nilpotent.ket.dag().inner(&projector.ket).is_zero() {
                    Self::Nilpotent {
                        nilpotent: Nilpotent {
                            scalar: projector.scalar * nilpotent.scalar,
                            ket: nilpotent.ket,
                        },
                    }
                } else {
                    Self::Rank1 {
                        rank1: Rank1 {
                            scalar: nilpotent.scalar
                                * projector.scalar
                                * nilpotent.ket.dag().orthogonal().inner(&projector.ket),
                            ket: nilpotent.ket,
                            bra: projector.ket.dag(),
                        },
                    }
                }
            }
            (Self::Rank1 { rank1 }, Self::Projector { projector }) => Self::Rank1 {
                rank1: Rank1 {
                    scalar: projector.scalar * rank1.scalar * rank1.bra.inner(&projector.ket),
                    ket: rank1.ket,
                    bra: projector.ket.dag(),
                },
            },
            (Self::Rank2SVD { rank2 }, Self::Projector { projector }) => {
                let mut new_ket = projector.scalar
                    * (rank2.sigma1 * rank2.u1 * rank2.v1.inner(&projector.ket)
                        + rank2.sigma2 * rank2.v2 * rank2.u2.inner(&projector.ket))
                    .regularize();
                let new_scalar = new_ket.phase();
                new_ket = new_ket.normalize();
                if projector.ket == new_ket {
                    Self::Projector {
                        projector: Projector {
                            scalar: new_scalar,
                            ket: projector.ket,
                        },
                    }
                } else if projector.ket.dag().inner(&new_ket).is_zero() {
                    Self::Nilpotent {
                        nilpotent: Nilpotent {
                            scalar: new_scalar,
                            ket: new_ket,
                        },
                    }
                } else {
                    Self::Rank1 {
                        rank1: Rank1 {
                            scalar: new_scalar,
                            ket: new_ket,
                            bra: projector.ket.dag(),
                        },
                    }
                }
            }
            (Self::Pauli { pauli }, Self::Projector { projector }) => match pauli.idx {
                0 => Self::Projector {
                    projector: projector * pauli.scalar,
                },
                1 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: projector.scalar * pauli.scalar,
                        ket: KET_X0.dag().inner(&projector.ket) * KET_X0
                            - KET_X1.dag().inner(&projector.ket) * KET_X1,
                        bra: projector.ket.dag(),
                    },
                }
                .regularize(),
                2 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: projector.scalar * pauli.scalar,
                        ket: KET_Y0.dag().inner(&projector.ket) * KET_Y0
                            - KET_Y1.dag().inner(&projector.ket) * KET_Y1,
                        bra: projector.ket.dag(),
                    },
                }
                .regularize(),
                3 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: projector.scalar * pauli.scalar,
                        ket: KET_Z0.dag().inner(&projector.ket) * KET_Z0
                            - KET_Z1.dag().inner(&projector.ket) * KET_Z1,
                        bra: projector.ket.dag(),
                    },
                }
                .regularize(),
                _ => unreachable!(),
            },
            (
                Self::Nilpotent { nilpotent },
                Self::Nilpotent {
                    nilpotent: nilpotent2,
                },
            ) => {
                if nilpotent.ket == nilpotent2.ket {
                    Self::Rank0
                } else if nilpotent.ket.dag().inner(&nilpotent2.ket).is_zero() {
                    Self::Projector {
                        projector: Projector {
                            scalar: nilpotent.scalar * nilpotent2.scalar,
                            ket: nilpotent.ket,
                        },
                    }
                } else {
                    Self::Rank1 {
                        rank1: Rank1 {
                            scalar: nilpotent.scalar
                                * nilpotent2.scalar
                                * nilpotent.ket.dag().inner(&nilpotent2.ket),
                            ket: nilpotent.ket,
                            bra: nilpotent2.ket.dag(),
                        },
                    }
                }
            }
            (Self::Nilpotent { nilpotent }, Self::Rank1 { rank1 }) => Self::Rank1 {
                rank1: Rank1 {
                    scalar: nilpotent.scalar
                        * rank1.scalar
                        * nilpotent.ket.dag().orthogonal().inner(&rank1.ket),
                    ket: nilpotent.ket,
                    bra: rank1.bra,
                },
            }
            .regularize(),
            (Self::Rank1 { rank1 }, Self::Nilpotent { nilpotent }) => Self::Rank1 {
                rank1: Rank1 {
                    scalar: nilpotent.scalar * rank1.scalar * rank1.bra.inner(&nilpotent.ket),
                    ket: rank1.ket,
                    bra: nilpotent.ket.dag().orthogonal(),
                },
            }
            .regularize(),
            (Self::Nilpotent { nilpotent }, Self::Rank2SVD { rank2 }) => {
                let mut new_bra = nilpotent.scalar
                    * (rank2.sigma1 * rank2.v1 * nilpotent.ket.dag().orthogonal().inner(&rank2.u1)
                        + rank2.sigma2
                            * rank2.v2
                            * nilpotent.ket.dag().orthogonal().inner(&rank2.u2))
                    .regularize();
                let new_scalar = new_bra.phase();
                new_bra = new_bra.normalize();
                if nilpotent.ket == new_bra.dag() {
                    Self::Projector {
                        projector: Projector {
                            scalar: new_scalar,
                            ket: nilpotent.ket,
                        },
                    }
                } else if new_bra.inner(&nilpotent.ket).is_zero() {
                    Self::Nilpotent {
                        nilpotent: Nilpotent {
                            scalar: new_scalar,
                            ket: nilpotent.ket,
                        },
                    }
                } else {
                    Self::Rank1 {
                        rank1: Rank1 {
                            scalar: new_scalar,
                            ket: nilpotent.ket,
                            bra: new_bra,
                        },
                    }
                }
            }
            (Self::Rank2SVD { rank2 }, Self::Nilpotent { nilpotent }) => {
                let mut new_ket = nilpotent.scalar
                    * (rank2.sigma1 * rank2.u1 * rank2.v1.inner(&nilpotent.ket)
                        + rank2.sigma2 * rank2.v2 * rank2.u2.inner(&nilpotent.ket))
                    .regularize();
                let new_scalar = new_ket.phase();
                new_ket = new_ket.normalize();
                if nilpotent.ket.orthogonal() == new_ket {
                    Self::Projector {
                        projector: Projector {
                            scalar: new_scalar,
                            ket: new_ket,
                        },
                    }
                } else if nilpotent.ket.dag().orthogonal().inner(&new_ket).is_zero() {
                    Self::Nilpotent {
                        nilpotent: Nilpotent {
                            scalar: new_scalar,
                            ket: new_ket,
                        },
                    }
                } else {
                    Self::Rank1 {
                        rank1: Rank1 {
                            scalar: new_scalar,
                            ket: new_ket,
                            bra: nilpotent.ket.dag().orthogonal(),
                        },
                    }
                }
            }
            (Self::Nilpotent { nilpotent }, Self::Pauli { pauli }) => match pauli.idx {
                0 => Self::Nilpotent {
                    nilpotent: nilpotent * pauli.scalar,
                },
                1 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: nilpotent.scalar * pauli.scalar,
                        ket: nilpotent.ket,
                        bra: nilpotent.ket.dag().orthogonal().inner(&KET_X0) * KET_X0.dag()
                            - nilpotent.ket.dag().orthogonal().inner(&KET_X1) * KET_X1.dag(),
                    },
                }
                .regularize(),
                2 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: nilpotent.scalar * pauli.scalar,
                        ket: nilpotent.ket,
                        bra: nilpotent.ket.dag().orthogonal().inner(&KET_Y0) * KET_Y0.dag()
                            - nilpotent.ket.dag().orthogonal().inner(&KET_Y1) * KET_Y1.dag(),
                    },
                }
                .regularize(),
                3 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: nilpotent.scalar * pauli.scalar,
                        ket: nilpotent.ket,
                        bra: nilpotent.ket.dag().orthogonal().inner(&KET_Z0) * KET_Z0.dag()
                            - nilpotent.ket.dag().orthogonal().inner(&KET_Z1) * KET_Z1.dag(),
                    },
                }
                .regularize(),
                _ => unreachable!(),
            },
            (Self::Pauli { pauli }, Self::Nilpotent { nilpotent }) => match pauli.idx {
                0 => Self::Nilpotent {
                    nilpotent: nilpotent * pauli.scalar,
                },
                1 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: nilpotent.scalar * pauli.scalar,
                        ket: KET_X0.dag().inner(&nilpotent.ket) * KET_X0
                            - KET_X1.dag().inner(&nilpotent.ket) * KET_X1,
                        bra: nilpotent.ket.dag().orthogonal(),
                    },
                }
                .regularize(),
                2 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: nilpotent.scalar * pauli.scalar,
                        ket: KET_Y0.dag().inner(&nilpotent.ket) * KET_Y0
                            - KET_Y1.dag().inner(&nilpotent.ket) * KET_Y1,
                        bra: nilpotent.ket.dag().orthogonal(),
                    },
                }
                .regularize(),
                3 => Self::Rank1 {
                    rank1: Rank1 {
                        scalar: nilpotent.scalar * pauli.scalar,
                        ket: KET_Z0.dag().inner(&nilpotent.ket) * KET_Z0
                            - KET_Z1.dag().inner(&nilpotent.ket) * KET_Z1,
                        bra: nilpotent.ket.dag().orthogonal(),
                    },
                }
                .regularize(),
                _ => unreachable!(),
            },
            (Self::Rank1 { rank1 }, Self::Rank1 { rank1: rk1 }) => {
                Self::Rank1 { rank1: rank1 * rk1 }
            }
            (Self::Rank1 { rank1 }, Self::Rank2SVD { rank2 }) => Self::Rank1 {
                rank1: rank1 * rank2,
            },
            (Self::Rank1 { rank1 }, Self::Pauli { pauli }) => Self::Rank1 {
                rank1: rank1 * pauli,
            },
            (Self::Rank2SVD { rank2 }, Self::Rank1 { rank1 }) => Self::Rank1 {
                rank1: rank2 * rank1,
            },
            (Self::Pauli { pauli }, Self::Rank1 { rank1 }) => Self::Rank1 {
                rank1: pauli * rank1,
            },
            (Self::Rank2SVD { rank2 }, Self::Rank2SVD { rank2: rk2 }) => {
                Self::Rank2SVD { rank2: rank2 * rk2 }
            }
            (Self::Rank2SVD { rank2 }, Self::Pauli { pauli }) => Self::Rank2SVD {
                rank2: rank2 * pauli,
            },
            (Self::Pauli { pauli }, Self::Rank2SVD { rank2 }) => Self::Rank2SVD {
                rank2: pauli * rank2,
            },
            (Self::Pauli { pauli }, Self::Pauli { pauli: p2 }) => Self::Pauli { pauli: pauli * p2 },
        }
    }
}

impl Mul<Rank2SVD> for Rank1 {
    type Output = Rank1;
    fn mul(self, rhs: Rank2SVD) -> Self::Output {
        let rk1 = self.regularize();
        let rk2 = rhs.regularize();
        Rank1 {
            scalar: ONE,
            ket: rk1.ket,
            bra: rk1.scalar
                * (rk1.bra.inner(&rk2.u1) * rk2.v1 * rk2.sigma1
                    + rk1.bra.inner(&rk2.u2) * rk2.v2 * rk2.sigma2),
        }
        .regularize()
    }
}

impl Mul<Rank1> for Rank2SVD {
    type Output = Rank1;
    fn mul(self, rhs: Rank1) -> Self::Output {
        let rk1 = rhs.regularize();
        let rk2 = self.regularize();
        Rank1 {
            scalar: ONE,
            ket: (rk2.sigma1 * rk2.u1 * rk2.v1.inner(&rk1.ket)
                + rk2.sigma2 * rk2.u2 * rk2.v2.inner(&rk1.ket))
                * rk1.scalar,
            bra: rk1.bra,
        }
        .regularize()
    }
}
impl Mul<Rank1> for Pauli {
    type Output = Rank1;
    fn mul(self, rhs: Rank1) -> Self::Output {
        Rank1 {
            scalar: rhs.scalar.regularize(),
            ket: self * rhs.ket,
            bra: rhs.bra,
        }
        .regularize()
    }
}

impl Mul<Pauli> for Rank1 {
    type Output = Rank1;
    fn mul(self, rhs: Pauli) -> Self::Output {
        Rank1 {
            scalar: self.scalar.regularize(),
            ket: self.ket,
            bra: self.bra * rhs,
        }
        .regularize()
    }
}

impl Mul<Rank2SVD> for Pauli {
    type Output = Rank2SVD;
    fn mul(self, rhs: Rank2SVD) -> Self::Output {
        Rank2SVD {
            sigma1: rhs.sigma1.regularize(),
            sigma2: rhs.sigma2.regularize(),
            u1: self * rhs.u1,
            u2: self * rhs.u2,
            v1: rhs.v1,
            v2: rhs.v2,
        }
        .regularize()
    }
}

impl Mul<Pauli> for Rank2SVD {
    type Output = Rank2SVD;
    fn mul(self, rhs: Pauli) -> Self::Output {
        Rank2SVD {
            sigma1: self.sigma1.regularize(),
            sigma2: self.sigma2.regularize(),
            u1: self.u1,
            u2: self.u2,
            v1: self.v1 * rhs,
            v2: self.v2 * rhs,
        }
        .regularize()
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;

    #[test]
    fn test_ket_creation() {
        let ket = State::Ket {
            c0: Cpx::Real { re: 1.0 },
            c1: Cpx::Real { re: 2.0 },
            phase: Cpx::Real { re: 1.0 },
        };

        match ket {
            State::Ket { c0, c1, phase } => {
                match c0 {
                    Cpx::Real { re } => assert_eq!(re, 1.0),
                    _ => panic!("Incorrect value for c0 in Ket."),
                }
                match c1 {
                    Cpx::Real { re } => assert_eq!(re, 2.0),
                    _ => panic!("Incorrect value for c1 in Ket."),
                }
                match phase {
                    Cpx::Real { re } => assert_eq!(re, 1.0),
                    _ => panic!("Incorrect phase value in Ket."),
                }
            }
            _ => panic!("Failed to create Ket state."),
        }
    }

    #[test]
    fn test_bra_creation() {
        let bra = State::Bra {
            c0: Cpx::Real { re: 3.0 },
            c1: Cpx::Real { re: 4.0 },
            phase: Cpx::Real { re: 1.0 },
        };

        match bra {
            State::Bra { c0, c1, phase } => {
                match c0 {
                    Cpx::Real { re } => assert_eq!(re, 3.0),
                    _ => panic!("Incorrect value for c0 in Bra."),
                }
                match c1 {
                    Cpx::Real { re } => assert_eq!(re, 4.0),
                    _ => panic!("Incorrect value for c1 in Bra."),
                }
                match phase {
                    Cpx::Real { re } => assert_eq!(re, 1.0),
                    _ => panic!("Incorrect phase value in Bra."),
                }
            }
            _ => panic!("Failed to create Bra state."),
        }
    }

    #[test]
    fn test_bra_ket_inner_product() {
        let ket = State::Ket {
            c0: Cpx::Real { re: 1.0 },
            c1: Cpx::Real { re: 2.0 },
            phase: Cpx::Real { re: 1.0 },
        };
        let bra = State::Bra {
            c0: Cpx::Real { re: 3.0 },
            c1: Cpx::Real { re: 4.0 },
            phase: Cpx::Real { re: 1.0 },
        };

        let result = bra.inner(&ket);

        match result {
            Cpx::Real { re } => assert_eq!(re, 11.0), // Inner product should be 1*3 + 2*4
            _ => panic!("Inner product failed."),
        }
    }

    #[test]
    fn test_ket_negation() {
        let ket = State::Ket {
            c0: Cpx::Real { re: 3.0 },
            c1: Cpx::Real { re: 4.0 },
            phase: Cpx::Real { re: 1.0 },
        };

        // Applying regularization to the state
        let regularized_ket = ket.regularize();
        let result = -regularized_ket; // Negate the regularized state

        match result {
            State::Ket { c0, c1, phase } => {
                // Check if the normalization of c0, c1 is correct
                match c0 {
                    Cpx::Real { re } => assert_eq!(re, 0.6), // Expecting normalized value
                    _ => panic!("Negation failed for c0 in Ket."),
                }
                match c1 {
                    Cpx::Real { re } => assert_eq!(re, 0.8), // Expecting normalized value
                    _ => panic!("Negation failed for c1 in Ket."),
                }
                match phase {
                    Cpx::Real { re } => assert_eq!(re, -5.0), // Sign unchanged after negation
                    _ => panic!("Negation failed for phase in Ket."),
                }
            }
            _ => panic!("Failed to negate Ket state."),
        }
    }

    #[test]
    fn test_bra_negation() {
        let bra = State::Bra {
            c0: Cpx::Real { re: 3.0 },
            c1: Cpx::Real { re: 4.0 },
            phase: Cpx::Real { re: 1.0 },
        };
        let regularized_bra = bra.regularize();
        let result = -regularized_bra; // Negate the regularized state

        match result {
            State::Bra { c0, c1, phase } => {
                match c0 {
                    Cpx::Real { re } => assert_eq!(re, 0.6),
                    _ => panic!("Negation failed for c0 in Bra."),
                }
                match c1 {
                    Cpx::Real { re } => assert_eq!(re, 0.8),
                    _ => panic!("Negation failed for c1 in Bra."),
                }
                match phase {
                    Cpx::Real { re } => assert_eq!(re, -5.0),
                    _ => panic!("Negation failed for phase in Bra."),
                }
            }
            _ => panic!("Failed to negate Bra state."),
        }
    }

    #[test]
    fn test_ket_addition() {
        let ket1 = State::Ket {
            c0: Cpx::Real { re: 1.0 },
            c1: Cpx::Real { re: 2.0 },
            phase: Cpx::Real { re: 1.0 },
        };
        let ket2 = State::Ket {
            c0: Cpx::Real { re: 2.0 },
            c1: Cpx::Real { re: 2.0 },
            phase: Cpx::Real { re: 1.0 },
        };

        let result = ket1 + ket2;

        match result {
            State::Ket { c0, c1, phase } => {
                match c0 {
                    Cpx::Real { re } => assert_eq!(re, 0.6),
                    _ => panic!("Addition failed for c0 in Ket."),
                }
                match c1 {
                    Cpx::Real { re } => assert_eq!(re, 0.8),
                    _ => panic!("Addition failed for c1 in Ket."),
                }
                match phase {
                    Cpx::Real { re } => assert_eq!(re, 5.0),
                    _ => panic!("Addition failed for phase in Ket."),
                }
            }
            _ => panic!("Failed to add Ket states."),
        }
    }

    #[test]
    fn test_bra_ket_inner_product_zero() {
        let ket = State::Ket {
            c0: Cpx::Real { re: 1.0 },
            c1: Cpx::Real { re: 0.0 },
            phase: Cpx::Real { re: 1.0 },
        };
        let bra = State::Bra {
            c0: Cpx::Real { re: 0.0 },
            c1: Cpx::Real { re: 1.0 },
            phase: Cpx::Real { re: 1.0 },
        };

        let result = bra.inner(&ket);

        match result {
            Cpx::Zero {} => {} // The inner product should be 0
            _ => panic!("Inner product failed, expected 0."),
        }
    }

    #[test]
    fn test_state_normalization() {
        let ket = State::Ket {
            c0: Cpx::Real { re: 3.0 },
            c1: Cpx::Real { re: -4.0 },
            phase: Cpx::Real { re: 1.0 },
        };

        let normalized_ket = ket.normalize();

        match normalized_ket {
            State::Ket { c0, c1, phase } => {
                match c0 {
                    Cpx::Real { re } => assert_eq!(re, 0.6),
                    _ => panic!("Normalization failed for c0 in Ket."),
                }
                match c1 {
                    Cpx::Real { re } => assert_eq!(re, -0.8),
                    _ => panic!("Normalization failed for c1 in Ket."),
                }
                match phase {
                    Cpx::One {} => {}
                    _ => panic!("Normalization failed for phase in Ket."),
                }
            }
            _ => panic!("Failed to normalize Ket state."),
        }
    }
}

#[test]
fn test_rank1_creation() {
    let scalar = ONE;
    let ket = State::Ket {
        c0: ONE,
        c1: ZERO,
        phase: ONE,
    };
    let bra = ket.dag(); // Assuming dag() gives the conjugate transpose

    let rank1 = Rank1 { scalar, ket, bra };
    assert_eq!(rank1.scalar, ONE);
    assert_eq!(rank1.ket, ket);
    assert_eq!(rank1.bra, bra);
    println!("{:?}", rank1);
}

#[test]
fn test_rank1_clone_and_copy() {
    let rank1 = Rank1 {
        scalar: Cpx::Real { re: 2.0 },
        ket: State::Ket {
            c0: Cpx::J {},
            c1: Cpx::NegJ {},
            phase: Cpx::One {},
        },
        bra: State::Bra {
            c0: Cpx::J {},
            c1: Cpx::NegJ {},
            phase: Cpx::One {},
        },
    };

    let rank1_clone = rank1.clone();
    let rank1_copy = rank1; // Should work because of Copy trait

    assert_eq!(rank1, rank1_clone);
    assert_eq!(rank1, rank1_copy);
}

#[test]
fn test_rank1_arithmetic() {
    let r1 = Rank1 {
        scalar: Cpx::Real { re: 3.0 },
        ket: State::Ket {
            c0: Cpx::Real { re: 1.0 },
            c1: Cpx::Real { re: 2.0 },
            phase: Cpx::One {},
        },
        bra: State::Bra {
            c0: Cpx::Real { re: 1.0 },
            c1: Cpx::Real { re: -2.0 },
            phase: Cpx::One {},
        },
    };

    let r2 = Rank1 {
        scalar: Cpx::Real { re: -1.0 },
        ket: State::Ket {
            c0: Cpx::J {},
            c1: Cpx::NegJ {},
            phase: Cpx::One {},
        },
        bra: State::Bra {
            c0: Cpx::J {},
            c1: Cpx::NegJ {},
            phase: Cpx::One {},
        },
    };

    let product = r1 * r2;

    println!("Product: {:?}", product);
}

#[test]
fn test_rank1_mul_state() {
    let scalar = Cpx::One {};
    let ket = State::Ket {
        c0: Cpx::One {},
        c1: Cpx::Zero {},
        phase: Cpx::One {},
    };
    let bra = State::Bra {
        c0: Cpx::One {},
        c1: Cpx::Zero {},
        phase: Cpx::One {},
    };
    let rank1 = Rank1 { scalar, ket, bra };

    let test_state = State::Ket {
        c0: Cpx::Real { re: 0.6 },
        c1: Cpx::Real { re: 0.8 },
        phase: Cpx::One {},
    };

    let result = rank1 * test_state; // Assuming you've implemented Mul for Rank1 * State

    let expected = State::Ket {
        c0: Cpx::One {},
        c1: Cpx::Zero {},
        phase: Cpx::Real { re: 0.6 },
    };

    assert_eq!(result, expected);
    println!("{:?}", result); // Debug output to verify correctness
}
#[test]
fn test_rank1_mul_cpx() {
    let scalar = Cpx::One {};
    let ket = State::Ket {
        c0: Cpx::One {},
        c1: Cpx::Zero {},
        phase: Cpx::One {},
    };
    let bra = State::Bra {
        c0: Cpx::One {},
        c1: Cpx::Zero {},
        phase: Cpx::One {},
    };
    let rank1 = Rank1 { scalar, ket, bra };

    let multiplier = Cpx::Real { re: 2.0 };
    let result = rank1 * multiplier; // Assuming Mul<Cpx> is implemented for Rank1

    let expected = Rank1 {
        scalar: Cpx::Real { re: 2.0 },
        ket,
        bra,
    };

    assert_eq!(result, expected);
    println!("{:?}", result); // Debug output to verify correctness
}

#[test]
fn test_rank2svd_mul_state() {
    let rank2 = Rank2SVD {
        sigma1: Cpx::Real { re: 2.0 },
        sigma2: Cpx::Real { re: 1.0 },
        u1: State::Ket {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        u2: State::Ket {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
        v1: State::Bra {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        v2: State::Bra {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
    };

    let state = State::Ket {
        c0: Cpx::One {},
        c1: Cpx::Zero {},
        phase: Cpx::One {},
    };
    let result = rank2 * state;

    let expected = State::Ket {
        c0: ONE,
        c1: Cpx::Zero {},
        phase: Cpx::Real { re: 2.0 },
    };
    assert_eq!(result, expected);
    println!("{:?}", result);
}

#[test]
fn test_rank2svd_mul_rank1() {
    let rank2 = Rank2SVD {
        sigma1: Cpx::Real { re: 2.0 },
        sigma2: Cpx::Real { re: 1.0 },
        u1: State::Ket {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        u2: State::Ket {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
        v1: State::Bra {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        v2: State::Bra {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
    };

    let rank1 = Rank1 {
        scalar: Cpx::Real { re: 3.0 },
        ket: State::Ket {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        bra: State::Bra {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
    };

    let result = rank2 * rank1;

    let expected = Rank1 {
        scalar: Cpx::Real { re: 6.0 }, // 2.0 * 3.0
        ket: State::Ket {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        bra: State::Bra {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
    };

    assert_eq!(result, expected);
    println!("{:?}", result);
}

#[test]
fn test_rank2svd_mul_rank2svd() {
    let rank2_a = Rank2SVD {
        sigma1: Cpx::Real { re: 2.0 },
        sigma2: Cpx::Real { re: 1.0 },
        u1: State::Ket {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        u2: State::Ket {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
        v1: State::Bra {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        v2: State::Bra {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
    };

    let rank2_b = Rank2SVD {
        sigma1: Cpx::Real { re: 3.0 },
        sigma2: Cpx::Real { re: 4.0 },
        u1: State::Ket {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        u2: State::Ket {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
        v1: State::Bra {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        v2: State::Bra {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
    };

    let result = rank2_a * rank2_b;

    let expected = Rank2SVD {
        sigma1: Cpx::Real { re: 6.0 }, // 2.0 * 3.0
        sigma2: Cpx::Real { re: 4.0 }, // 1.0 * 4.0
        u1: State::Ket {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        u2: State::Ket {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
        v1: State::Bra {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        v2: State::Bra {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
    };

    assert_eq!(result, expected);
    println!("{:?}", result);
}

#[test]
fn test_rank2svd_mul_cpx() {
    let rank2 = Rank2SVD {
        sigma1: Cpx::Real { re: 2.0 },
        sigma2: Cpx::Real { re: 1.0 },
        u1: State::Ket {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        u2: State::Ket {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
        v1: State::Bra {
            c0: Cpx::One {},
            c1: Cpx::Zero {},
            phase: Cpx::One {},
        },
        v2: State::Bra {
            c0: Cpx::Zero {},
            c1: Cpx::One {},
            phase: Cpx::One {},
        },
    };

    let multiplier = Cpx::Real { re: 3.0 };
    let result = rank2 * multiplier;

    let expected = Rank2SVD {
        sigma1: Cpx::Real { re: 6.0 },
        sigma2: Cpx::Real { re: 3.0 },
        u1: rank2.u1,
        u2: rank2.u2,
        v1: rank2.v1,
        v2: rank2.v2,
    };

    assert_eq!(result, expected);
    println!("{:?}", result);
}

#[test]
fn test_pauli_creation() {
    let pauli_x = Pauli {
        idx: 1,
        scalar: Cpx::One {},
    };
    let pauli_y = Pauli {
        idx: 2,
        scalar: Cpx::Real { re: -1.0 },
    };
    let pauli_z = Pauli {
        idx: 3,
        scalar: Cpx::Imag { im: 1.0 },
    };

    assert_eq!(pauli_x.idx, 1);
    assert_eq!(pauli_y.idx, 2);
    assert_eq!(pauli_z.idx, 3);

    println!("{:?}", pauli_x);
    println!("{:?}", pauli_y);
    println!("{:?}", pauli_z);
}

#[test]
fn test_pauli_mul_pauli() {
    let pauli_x = Pauli {
        idx: 1,
        scalar: Cpx::One {},
    };
    let pauli_y = Pauli {
        idx: 2,
        scalar: Cpx::One {},
    };
    let pauli_z = Pauli {
        idx: 3,
        scalar: Cpx::One {},
    };
    let identity = Pauli {
        idx: 0,
        scalar: Cpx::One {},
    };

    // X * X = I
    assert_eq!(pauli_x * pauli_x, identity);
    // Y * Y = I
    assert_eq!(pauli_y * pauli_y, identity);
    // Z * Z = I
    assert_eq!(pauli_z * pauli_z, identity);
    // X * Y = iZ
    assert_eq!(
        pauli_x * pauli_y,
        Pauli {
            idx: 3,
            scalar: Cpx::Imag { im: 1.0 }
        }
    );
    // Y * X = -iZ
    assert_eq!(
        pauli_y * pauli_x,
        Pauli {
            idx: 3,
            scalar: Cpx::Imag { im: -1.0 }
        }
    );

    println!("{:?}", pauli_x * pauli_y);
    println!("{:?}", pauli_y * pauli_x);
}

#[test]
fn test_pauli_mul_cpx() {
    let pauli_x = Pauli {
        idx: 1,
        scalar: Cpx::One {},
    };
    let multiplier = Cpx::Real { re: 2.0 };

    let result = pauli_x * multiplier;
    let expected = Pauli {
        idx: 1,
        scalar: Cpx::Real { re: 2.0 },
    };

    assert_eq!(result, expected);
    println!("{:?}", result);
}

#[test]
fn test_pauli_identity() {
    let identity = Pauli {
        idx: 0,
        scalar: Cpx::One {},
    };
    let pauli_x = Pauli {
        idx: 1,
        scalar: Cpx::One {},
    };

    assert_eq!(identity * pauli_x, pauli_x);
    assert_eq!(pauli_x * identity, pauli_x);

    println!("{:?}", identity * pauli_x);
    println!("{:?}", pauli_x * identity);
}

#[test]
fn test_pauli_negation() {
    let pauli_x = Pauli {
        idx: 1,
        scalar: Cpx::One {},
    };
    let neg_pauli_x = -pauli_x;

    let expected = Pauli {
        idx: 1,
        scalar: Cpx::NegOne {},
    };
    assert_eq!(neg_pauli_x, expected);

    println!("{:?}", neg_pauli_x);
}

#[test]
fn test_pauli_mul_state() {
    // Define a Pauli matrix (X, Y, Z)
    let pauli_x = Pauli {
        idx: 1,
        scalar: ONE,
    };
    let pauli_y = Pauli {
        idx: 2,
        scalar: ONE,
    };
    let pauli_z = Pauli {
        idx: 3,
        scalar: ONE,
    };

    // Define a state (Ket and Bra)
    let ket = State::Ket {
        c0: ONE,
        c1: ZERO,
        phase: ONE,
    };
    let bra = State::Bra {
        c0: ONE,
        c1: ZERO,
        phase: ONE,
    };

    // Pauli X on Ket
    let result_ket_x = pauli_x * ket;
    let expected_ket_x = State::Ket {
        c0: ZERO,
        c1: ONE,
        phase: ONE,
    };
    assert_eq!(result_ket_x, expected_ket_x);

    // Pauli X on Bra
    let result_bra_x = bra * pauli_x;
    let expected_bra_x = State::Bra {
        c0: ZERO,
        c1: ONE,
        phase: ONE,
    };
    assert_eq!(result_bra_x, expected_bra_x);

    // Pauli Y on Ket
    let result_ket_y = pauli_y * ket;
    let expected_ket_y = State::Ket {
        c0: ZERO,
        c1: ONE,
        phase: Cpx::Imag { im: 1.0 },
    };
    assert_eq!(result_ket_y, expected_ket_y);

    // Pauli Z on Ket
    let result_ket_z = pauli_z * ket;
    let expected_ket_z = State::Ket {
        c0: ONE,
        c1: ZERO,
        phase: ONE,
    };
    assert_eq!(result_ket_z, expected_ket_z);

    println!("{:?}", result_ket_x);
    println!("{:?}", result_bra_x);
    println!("{:?}", result_ket_y);
    println!("{:?}", result_ket_z);
}
