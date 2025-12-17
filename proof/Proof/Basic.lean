import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Normed.Module.Basic

noncomputable section

-- 设定环境：n维欧几里得空间
variable {n : Type*} [Fintype n] [DecidableEq n]

namespace W2Attn

/--
Structure representing a Gaussian Token with diagonal covariance.
Contains a mean vector `mu` and a standard deviation vector `sigma` (one per dimension).
This models a Gaussian with axis-aligned ellipsoidal uncertainty.
-/
structure GaussianToken (n : Type*) [Fintype n] where
  mu : EuclideanSpace ℝ n
  sigma : n → ℝ   -- Per-dimension standard deviation
  sigma_pos : ∀ i, 0 < sigma i  -- Positive per dimension

/--
Squared Wasserstein-2 distance between two Gaussian distributions with diagonal covariance.
For Gaussians with diagonal covariance matrices, W2² decomposes as:
  D² = Σᵢ (μ₁ᵢ - μ₂ᵢ)² + Σᵢ (√σ₁ᵢ - √σ₂ᵢ)²

This is the exact W2 distance when both Gaussians have diagonal covariance matrices.
-/
def w2_dist_sq (t1 t2 : GaussianToken n) : ℝ :=
  (∑ i, (t1.mu i - t2.mu i) ^ 2) +
  (∑ i, (Real.sqrt (t1.sigma i) - Real.sqrt (t2.sigma i)) ^ 2)

/-- W2 distance definition -/
def w2_dist (t1 t2 : GaussianToken n) : ℝ :=
  Real.sqrt (w2_dist_sq t1 t2)

end W2Attn
