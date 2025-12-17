import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Normed.Module.Basic

noncomputable section

-- 设定环境：n维欧几里得空间
variable {n : Type*} [Fintype n] [DecidableEq n]

namespace W2Attn

/-- 
Structure representing a Gaussian Token.
Contains a mean vector `mu` and a scalar standard deviation `sigma`.
-/
structure GaussianToken (n : Type*) [Fintype n] where
  mu : EuclideanSpace ℝ n
  sigma : ℝ
  sigma_nonneg : 0 ≤ sigma -- 物理约束：标准差非负

/-- 
Squared Wasserstein-2 distance between two Gaussian distributions with scalar variance.
D^2 = ||mu1 - mu2||^2 + (sigma1 - sigma2)^2
-/
def w2_dist_sq (t1 t2 : GaussianToken n) : ℝ :=
  ‖t1.mu - t2.mu‖ ^ 2 + (t1.sigma - t2.sigma) ^ 2

/-- W2 距离定义 -/
def w2_dist (t1 t2 : GaussianToken n) : ℝ :=
  Real.sqrt (w2_dist_sq t1 t2)

end W2Attn
