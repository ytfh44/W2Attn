import Proof.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2

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
  rw [norm_sub_rev t2.mu t1.mu]
  ring

/--
Helper: The W2 squared distance components are non-negative.
-/
theorem w2_dist_sq_nonneg (t1 t2 : GaussianToken n) : 0 ≤ w2_dist_sq t1 t2 := by
  unfold w2_dist_sq
  apply add_nonneg <;> apply sq_nonneg

/--
Property 3: Triangle Inequality.
We prove this using the Minkowski inequality for L2 norms:
  sqrt((a+c)² + (b+d)²) ≤ sqrt(a² + b²) + sqrt(c² + d²)
-/
theorem w2_triangle (t1 t2 t3 : GaussianToken n) :
    w2_dist t1 t3 ≤ w2_dist t1 t2 + w2_dist t2 t3 := by
  unfold w2_dist w2_dist_sq
  -- Extract the components
  set a := ‖t1.mu - t2.mu‖ with ha_def
  set b := t1.sigma - t2.sigma with hb_def
  set c := ‖t2.mu - t3.mu‖ with hc_def
  set d := t2.sigma - t3.sigma with hd_def
  have ha' : 0 ≤ a := norm_nonneg _
  have hc' : 0 ≤ c := norm_nonneg _
  -- Triangle inequality for the mu component
  have h_mu : ‖t1.mu - t3.mu‖ ≤ a + c := by
    calc ‖t1.mu - t3.mu‖ = ‖(t1.mu - t2.mu) + (t2.mu - t3.mu)‖ := by congr 1; module
      _ ≤ ‖t1.mu - t2.mu‖ + ‖t2.mu - t3.mu‖ := norm_add_le _ _
      _ = a + c := rfl
  -- Sigma component addition
  have h_sigma : t1.sigma - t3.sigma = b + d := by ring
  rw [h_sigma]
  -- First reduce: sqrt(‖μ13‖² + (b+d)²) ≤ sqrt((a+c)² + (b+d)²)
  have lhs_le : ‖t1.mu - t3.mu‖ ^ 2 + (b + d) ^ 2 ≤ (a + c) ^ 2 + (b + d) ^ 2 := by
    have hmu_nn : 0 ≤ ‖t1.mu - t3.mu‖ := norm_nonneg _
    have hac_nn : 0 ≤ a + c := add_nonneg ha' hc'
    have h1 : ‖t1.mu - t3.mu‖ ^ 2 ≤ (a + c) ^ 2 := sq_le_sq' (by linarith) h_mu
    linarith
  apply le_trans (Real.sqrt_le_sqrt lhs_le)
  -- Apply Minkowski: sqrt((a+c)² + (b+d)²) ≤ sqrt(a² + b²) + sqrt(c² + d²)
  have h1 : 0 ≤ a ^ 2 + b ^ 2 := add_nonneg (sq_nonneg _) (sq_nonneg _)
  have h2 : 0 ≤ c ^ 2 + d ^ 2 := add_nonneg (sq_nonneg _) (sq_nonneg _)
  have hsum : 0 ≤ Real.sqrt (a ^ 2 + b ^ 2) + Real.sqrt (c ^ 2 + d ^ 2) :=
    add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  rw [← Real.sqrt_sq hsum]
  apply Real.sqrt_le_sqrt
  -- Need: (a+c)² + (b+d)² ≤ (sqrt(a² + b²) + sqrt(c² + d²))²
  -- Expand RHS: (sqrt(a²+b²))² + 2*sqrt(a²+b²)*sqrt(c²+d²) + (sqrt(c²+d²))²
  --           = a² + b² + c² + d² + 2*sqrt((a²+b²)(c²+d²))
  have rhs_expand : (Real.sqrt (a ^ 2 + b ^ 2) + Real.sqrt (c ^ 2 + d ^ 2)) ^ 2 =
                    a^2 + b^2 + c^2 + d^2 + 2 * Real.sqrt ((a^2 + b^2) * (c^2 + d^2)) := by
    rw [add_sq, Real.sq_sqrt h1, Real.sq_sqrt h2, Real.sqrt_mul h1]
    ring
  rw [rhs_expand]
  -- Expand LHS: (a+c)² + (b+d)² = a² + c² + 2ac + b² + d² + 2bd = a² + b² + c² + d² + 2(ac+bd)
  have lhs_expand : (a + c) ^ 2 + (b + d) ^ 2 = a^2 + b^2 + c^2 + d^2 + 2*(a*c + b*d) := by ring
  rw [lhs_expand]
  -- Need: 2*(a*c + b*d) ≤ 2*sqrt((a²+b²)(c²+d²))
  gcongr
  -- Cauchy-Schwarz: a*c + b*d ≤ sqrt((a²+b²)(c²+d²))
  have hprod_nn : 0 ≤ (a^2 + b^2) * (c^2 + d^2) := mul_nonneg h1 h2
  have key : (a*c + b*d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) := by nlinarith [sq_nonneg (a*d - b*c)]
  -- |a*c + b*d| ≤ sqrt((a²+b²)(c²+d²)) follows from squaring
  have h_abs : |a*c + b*d| ≤ Real.sqrt ((a^2 + b^2) * (c^2 + d^2)) := by
    have h1 : |a*c + b*d| = Real.sqrt ((a*c + b*d)^2) := (Real.sqrt_sq_eq_abs _).symm
    rw [h1]
    exact Real.sqrt_le_sqrt key
  exact le_trans (le_abs_self _) h_abs

end W2Attn
