import Proof.Basic

noncomputable section

variable {n : Type*} [Fintype n] [DecidableEq n]

namespace W2Attn

/--
Theorem: Degeneracy under fixed sigma.
If two tokens have the same sigma vector, W2 distance reduces to Euclidean distance.
-/
theorem w2_reduces_to_euclidean (t1 t2 : GaussianToken n) (h : t1.sigma = t2.sigma) :
  w2_dist_sq t1 t2 = ∑ i, (t1.mu i - t2.mu i) ^ 2 := by
  unfold w2_dist_sq
  have sigma_term_zero : ∑ i, (Real.sqrt (t1.sigma i) - Real.sqrt (t2.sigma i)) ^ 2 = 0 := by
    apply Finset.sum_eq_zero
    intro i _
    rw [h]
    ring
  linarith [sigma_term_zero]

/--
Theorem: Isotropic Degeneracy.
When all sigmas are equal (σ₁ᵢ = σ₂ᵢ = σ for all i), W2 reduces to L2 distance.
-/
theorem w2_isotropic (t1 t2 : GaussianToken n) (σ : ℝ) (hσ : 0 < σ)
  (h1 : ∀ i, t1.sigma i = σ) (h2 : ∀ i, t2.sigma i = σ) :
  w2_dist_sq t1 t2 = ∑ i, (t1.mu i - t2.mu i) ^ 2 := by
  apply w2_reduces_to_euclidean
  ext i
  rw [h1, h2]

end W2Attn
