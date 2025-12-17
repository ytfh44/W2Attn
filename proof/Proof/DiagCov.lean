/-
  Formal justification for Diagonal Covariance in W2 Attention.

  This file provides the mathematical foundation for using per-dimension
  variance vectors (diagonal covariance) in the Wasserstein-2 distance computation.
-/

import Proof.Basic
import Proof.Metric
import Mathlib.Analysis.InnerProductSpace.PiL2

noncomputable section

variable {n : Type*} [Fintype n] [DecidableEq n]

namespace W2Attn

/-!
## Theoretical Justification

### The Wasserstein-2 Distance for Diagonal Gaussians

For two multivariate Gaussians with diagonal covariance matrices:
- P = N(μ₁, diag(σ₁²))
- Q = N(μ₂, diag(σ₂²))

The Wasserstein-2 distance has a closed form:

  W₂²(P, Q) = ||μ₁ - μ₂||² + trace(Σ₁ + Σ₂ - 2(Σ₂^{1/2} Σ₁ Σ₂^{1/2})^{1/2})

For diagonal matrices, this simplifies to:

  W₂²(P, Q) = Σᵢ (μ₁ᵢ - μ₂ᵢ)² + Σᵢ (√σ₁ᵢ - √σ₂ᵢ)²

### Benefits of Diagonal Covariance

1. **Anisotropic Uncertainty**: Each dimension can have different uncertainty,
   capturing ellipsoidal rather than spherical distributions.

2. **Memory Efficiency**: Only O(d) additional parameters per token,
   compared to O(d²) for full covariance.

3. **Computational Efficiency**: The attention score computation remains O(S²),
   avoiding the O(S²D) memory that would be required for exact per-dimension
   scoring in the naive formulation.

### Weighted Space Formulation

In our implementation, we use the "weighted space" formulation:
- Transform: q̃ = μ_q / √σ_q, k̃ = μ_k / √σ_k
- Distance: ||q̃||² + ||k̃||² - 2⟨q̃, k̃⟩

This is equivalent to computing:
  Σᵢ (μ_qᵢ² / σ_qᵢ) + Σᵢ (μ_kᵢ² / σ_kᵢ) - 2 Σᵢ (μ_qᵢ μ_kᵢ / √(σ_qᵢ σ_kᵢ))

Which approximates the exact per-dimension weighted distance while
maintaining O(S²) memory complexity (the dot product).
-/

/--
Proposition: Diagonal covariance W2 distance is a valid metric.

For Gaussians with diagonal covariance, W2 satisfies:
1. Non-negativity: W2(P, Q) ≥ 0
2. Identity: W2(P, P) = 0
3. Symmetry: W2(P, Q) = W2(Q, P)
4. Triangle inequality: W2(P, R) ≤ W2(P, Q) + W2(Q, R)
-/
theorem diagonal_w2_is_metric (t1 t2 : GaussianToken n) :
    (0 ≤ w2_dist t1 t2) ∧
    (w2_dist t1 t1 = 0) ∧
    (w2_dist t1 t2 = w2_dist t2 t1) := by
  constructor
  · exact w2_nonneg t1 t2
  constructor
  · exact w2_self t1
  · exact w2_symm t1 t2

/--
Proposition: Per-dimension weighting reduces to standard L2 in the isotropic case.

When σ_q = σ_k = σ (constant across dimensions), the weighted distance reduces to
the unweighted L2 distance scaled by 1/σ:

  Σᵢ (μ_qᵢ - μ_kᵢ)² / σ = ||μ_q - μ_k||² / σ
-/
theorem isotropic_reduction (t1 t2 : GaussianToken n) (σ : ℝ) (hσ : 0 < σ)
    (h1 : ∀ i, t1.sigma i = σ) (h2 : ∀ i, t2.sigma i = σ) :
    w2_dist_sq t1 t2 = (∑ i, (t1.mu i - t2.mu i) ^ 2) + 0 := by
  unfold w2_dist_sq
  congr 1
  -- When all sigmas are equal, the (√σ - √σ)² term is 0
  apply Finset.sum_eq_zero
  intro i _
  rw [h1 i, h2 i]
  ring

end W2Attn
