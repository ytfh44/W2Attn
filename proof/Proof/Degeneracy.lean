import Proof.Basic

noncomputable section

variable {n : Type*} [Fintype n] [DecidableEq n]

namespace W2Attn

/-- 
Theorem: Degeneracy under fixed sigma.
If two tokens have the same sigma, W2 distance reduces to Euclidean distanceSquared.
-/
theorem w2_reduces_to_euclidean (t1 t2 : GaussianToken n) (h : t1.sigma = t2.sigma) :
  w2_dist_sq t1 t2 = ‖t1.mu - t2.mu‖ ^ 2 := by
  unfold w2_dist_sq
  rw [h] -- Replace t1.sigma with t2.sigma
  simp -- (x - x)^2 = 0, eliminates sigma term

/-- 
Theorem: Point Mass Degeneracy.
When sigma is 0, the token degenerates to a standard point vector.
-/
theorem w2_zero_sigma (t1 t2 : GaussianToken n) 
  (h1 : t1.sigma = 0) (h2 : t2.sigma = 0) :
  w2_dist_sq t1 t2 = ‖t1.mu - t2.mu‖ ^ 2 := by
  apply w2_reduces_to_euclidean
  rw [h1, h2]

end W2Attn
