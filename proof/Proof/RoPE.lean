import Proof.Basic
import Mathlib.Analysis.Normed.Operator.LinearIsometry

noncomputable section

variable {n : Type*} [Fintype n] [DecidableEq n]

namespace W2Attn

open LinearIsometry

-- Rotary Positional Embedding (RoPE) as a Linear Isometry.
-- RoPE preserves vector length and distances between vectors.
variable (R : EuclideanSpace ℝ n →ₗᵢ[ℝ] EuclideanSpace ℝ n)

/--
Apply RoPE to a GaussianToken.
Rotates the mean `mu` only; standard deviation `sigma` is invariant.
-/
def apply_rope (t : GaussianToken n) : GaussianToken n :=
  { mu := R t.mu,
    sigma := t.sigma,
    sigma_nonneg := t.sigma_nonneg }

/--
Theorem: RoPE Invariance.
The Wasserstein-2 distance is invariant under RoPE rotation of the means.
W2(R(t1), R(t2)) = W2(t1, t2)
-/
theorem w2_invariant_under_rope (t1 t2 : GaussianToken n) :
  w2_dist (apply_rope R t1) (apply_rope R t2) = w2_dist t1 t2 := by
  -- Unfold definitions
  unfold w2_dist w2_dist_sq apply_rope
  simp only
  -- Use Isometry property: ||R x - R y|| = ||x - y||
  rw [← R.norm_map (t1.mu - t2.mu)]
  simp only [map_sub]

end W2Attn
