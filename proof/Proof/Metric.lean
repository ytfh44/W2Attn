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

def mu_dist (t1 t2 : GaussianToken n) : ℝ := dist t1.mu t2.mu

/-- Convert a function to EuclideanSpace -/
def toEuclidean (f : n → ℝ) : EuclideanSpace ℝ n :=
  (WithLp.equiv 2 (n → ℝ)).symm f

def sigma_dist (t1 t2 : GaussianToken n) : ℝ :=
  dist (toEuclidean (fun i => Real.sqrt (t1.sigma i)))
       (toEuclidean (fun i => Real.sqrt (t2.sigma i)))

theorem w2_dist_eq_hypot (t1 t2 : GaussianToken n) :
    w2_dist t1 t2 = Real.sqrt (mu_dist t1 t2 ^ 2 + sigma_dist t1 t2 ^ 2) := by
  unfold w2_dist w2_dist_sq mu_dist sigma_dist
  congr 1
  rw [dist_eq_norm, EuclideanSpace.norm_eq, dist_eq_norm, EuclideanSpace.norm_eq]
  simp only [Real.sq_sqrt (Finset.sum_nonneg (fun i _ => sq_nonneg _))]
  congr 1
  . congr; ext i; simp
  . congr; ext i; simp [toEuclidean, WithLp.equiv]

/-- Property 3: Triangle Inequality. -/
theorem w2_triangle (t1 t2 t3 : GaussianToken n) :
    w2_dist t1 t3 ≤ w2_dist t1 t2 + w2_dist t2 t3 := by
  rw [w2_dist_eq_hypot, w2_dist_eq_hypot, w2_dist_eq_hypot]
  let dmu12 := mu_dist t1 t2
  let dmu23 := mu_dist t2 t3
  let dmu13 := mu_dist t1 t3
  let dsigma12 := sigma_dist t1 t2
  let dsigma23 := sigma_dist t2 t3
  let dsigma13 := sigma_dist t1 t3

  have hmu : dmu13 ≤ dmu12 + dmu23 := dist_triangle t1.mu t2.mu t3.mu
  have hsigma : dsigma13 ≤ dsigma12 + dsigma23 := dist_triangle _ _ _

  let v12 : EuclideanSpace ℝ (Fin 2) := (WithLp.equiv 2 (Fin 2 → ℝ)).symm ![dmu12, dsigma12]
  let v23 : EuclideanSpace ℝ (Fin 2) := (WithLp.equiv 2 (Fin 2 → ℝ)).symm ![dmu23, dsigma23]
  let v_sum : EuclideanSpace ℝ (Fin 2) := (WithLp.equiv 2 (Fin 2 → ℝ)).symm ![dmu12 + dmu23, dsigma12 + dsigma23]
  let v13 : EuclideanSpace ℝ (Fin 2) := (WithLp.equiv 2 (Fin 2 → ℝ)).symm ![dmu13, dsigma13]

  have h_norm12 : ‖v12‖ = Real.sqrt (dmu12^2 + dsigma12^2) := by
    rw [EuclideanSpace.norm_eq]
    simp [Fin.sum_univ_two, v12]
  have h_norm23 : ‖v23‖ = Real.sqrt (dmu23^2 + dsigma23^2) := by
    rw [EuclideanSpace.norm_eq]
    simp [Fin.sum_univ_two, v23]
  have h_norm_sum : ‖v_sum‖ = Real.sqrt ((dmu12 + dmu23)^2 + (dsigma12 + dsigma23)^2) := by
    rw [EuclideanSpace.norm_eq]
    simp [Fin.sum_univ_two, v_sum]
  have h_norm13 : ‖v13‖ = Real.sqrt (dmu13^2 + dsigma13^2) := by
    rw [EuclideanSpace.norm_eq]
    simp [Fin.sum_univ_two, v13]

  rw [←h_norm13, ←h_norm12, ←h_norm23]

  calc ‖v13‖
    _ ≤ ‖v_sum‖ := by
      rw [h_norm13, h_norm_sum]
      apply Real.sqrt_le_sqrt
      gcongr
      . exact dist_nonneg
      . exact dist_nonneg
    _ ≤ ‖v12 + v23‖ := by
       have : v_sum = v12 + v23 := by
         ext i
         fin_cases i <;> simp [v_sum, v12, v23]
       rw [this]
    _ ≤ ‖v12‖ + ‖v23‖ := norm_add_le _ _

/-- Property 4: Identity of indiscernibles -/
theorem w2_self (t : GaussianToken n) : w2_dist t t = 0 := by
  unfold w2_dist w2_dist_sq
  simp

end W2Attn
