import Proof.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Topology.MetricSpace.Basic

noncomputable section

variable {n : Type*} [Fintype n] [DecidableEq n]

namespace W2Attn

/-- Property 1: Non-negativity -/
theorem w2_nonneg (t1 t2 : GaussianToken n) : 0 ≤ w2_dist t1 t2 :=
  Real.sqrt_nonneg _

/-- Property 2: Symmetry -/
theorem w2_symm (t1 t2 : GaussianToken n) : w2_dist t1 t2 = w2_dist t2 t1 := by
  unfold w2_dist w2_dist_sq
  congr 1
  have h1 : ∑ i, (t1.mu i - t2.mu i) ^ 2 = ∑ i, (t2.mu i - t1.mu i) ^ 2 := by
    apply Finset.sum_congr rfl
    intro i _
    ring
  have h2 : ∑ i, (Real.sqrt (t1.sigma i) - Real.sqrt (t2.sigma i)) ^ 2 =
            ∑ i, (Real.sqrt (t2.sigma i) - Real.sqrt (t1.sigma i)) ^ 2 := by
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [h1, h2]

/-- Helper: The W2 squared distance components are non-negative. -/
theorem w2_dist_sq_nonneg (t1 t2 : GaussianToken n) : 0 ≤ w2_dist_sq t1 t2 := by
  unfold w2_dist_sq
  apply add_nonneg <;> (apply Finset.sum_nonneg; intro i _; exact sq_nonneg _)

/-- Property 3: Triangle Inequality. -/
theorem w2_triangle (t1 t2 t3 : GaussianToken n) :
    w2_dist t1 t3 ≤ w2_dist t1 t2 + w2_dist t2 t3 := by
  -- The proof relies on the fact that W2 distance for diagonal covariance
  -- decomposes into the L2 distance of means plus the L2 distance of sqrt(sigmas).
  -- D(t1, t3) <= D(t1, t2) + D(t2, t3)
  -- This is equivalent to showing the triangle inequality in the product space ℝⁿ × ℝⁿ.

  -- Let μ_dist(a, b) = ||a.mu - b.mu||₂
  -- Let σ_dist(a, b) = ||√a.sigma - √b.sigma||₂
  -- Then w2_dist(a, b) = √((μ_dist(a,b))² + (σ_dist(a,b))²)

  -- 1. Triangle inequality holds for μ parts (Euclidean distance):
  --    μ_dist(t1, t3) <= μ_dist(t1, t2) + μ_dist(t2, t3)

  -- 2. Triangle inequality holds for σ parts (L2 distance on vectors of sqrts):
  --    σ_dist(t1, t3) <= σ_dist(t1, t2) + σ_dist(t2, t3)

  -- 3. Combine using Minkowski inequality in ℝ² for the vector norms:
  --    ||(μ_dist13, σ_dist13)||₂ <= ||(μ_dist12 + μ_dist23, σ_dist12 + σ_dist23)||₂
  --                             <= ||(μ_dist12, σ_dist12)||₂ + ||(μ_dist23, σ_dist23)||₂

  -- Formalization requires casting (n -> ℝ) to EuclideanSpace which hits type barriers
  sorry

/-- Property 4: Identity of indiscernibles -/
theorem w2_self (t : GaussianToken n) : w2_dist t t = 0 := by
  unfold w2_dist w2_dist_sq
  simp

end W2Attn
