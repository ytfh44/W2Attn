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
Rotates the mean `mu` only; variance `sigma` (per-dimension) is invariant.
RoPE is an isometric rotation that preserves the diagonal structure.
-/
def apply_rope (t : GaussianToken n) : GaussianToken n :=
  { mu := R t.mu,
    sigma := t.sigma,  -- Sigma unchanged (diagonal covariance preserved)
    sigma_pos := t.sigma_pos }

/--
Theorem: RoPE Invariance.
The Wasserstein-2 distance is invariant under RoPE rotation of the means.
W2(R(t1), R(t2)) = W2(t1, t2)

Key insight: For diagonal covariance, the W2 distance only depends on:
1. ||μ₁ - μ₂||² (preserved by isometry)
2. Σᵢ (√σ₁ᵢ - √σ₂ᵢ)² (unchanged since sigma is not rotated)
-/
theorem w2_invariant_under_rope (t1 t2 : GaussianToken n) :
    w2_dist (apply_rope R t1) (apply_rope R t2) = w2_dist t1 t2 := by
  unfold w2_dist w2_dist_sq apply_rope
  simp only
  -- The sigma terms are identical (unchanged by RoPE)
  congr 1
  congr 1
  -- For the mu terms: ||R(μ₁) - R(μ₂)||² = ||μ₁ - μ₂||²
  -- Use isometry property
  have h : ∀ i, (R t1.mu - R t2.mu) i = (R (t1.mu - t2.mu)) i := by
    intro i
    simp only [map_sub]
  -- The squared norms are equal under isometry
  have norm_eq : ‖R t1.mu - R t2.mu‖ = ‖t1.mu - t2.mu‖ := by
    rw [← R.norm_map (t1.mu - t2.mu)]
    simp only [map_sub]
  -- Convert from norm to sum of squares
  -- Convert from norm to sum of squares
  have norm_sq_eq : ‖R t1.mu - R t2.mu‖ ^ 2 = ‖t1.mu - t2.mu‖ ^ 2 := by rw [norm_eq]

  -- We need to transform norm squared to sum of components squared
  -- Lemma: ||x||^2 = sum (x_i)^2 for EuclideanSpace
  have norm_sq_to_sum (x : EuclideanSpace ℝ n) :
      ‖x‖ ^ 2 = ∑ i, (x i) ^ 2 := by
    rw [EuclideanSpace.norm_eq]
    rw [Real.sq_sqrt]
    · apply Finset.sum_congr rfl
      intro i _
      simp only [Real.norm_eq_abs, sq_abs]
    · apply Finset.sum_nonneg
      intro i _
      simp only [Real.norm_eq_abs, sq_nonneg]

  rw [norm_sq_to_sum, norm_sq_to_sum] at norm_sq_eq
  exact norm_sq_eq

end W2Attn
