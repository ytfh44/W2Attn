import Proof.Basic
import Proof.Metric

noncomputable section

variable {n : Type*} [Fintype n] [DecidableEq n]

namespace W2Attn

/--
ALiBi consists of adding a penalty linear in the distance between token positions.
In W2 Attention, the base score is -1/2 * W2^2.
ALiBi modifies this to: S_{ij} = -1/2 * W2^2 - m * |i - j|.
This theorem formalizes that this is equivalent to adding a transport cost penalty.
-/

/--
Definition of the Attention Score with ALiBi.
-/
def w2_score (t1 t2 : GaussianToken n) : ℝ :=
  -0.5 * (w2_dist_sq t1 t2)

def alibi_penalty (pos_i pos_j : ℕ) (m : ℝ) : ℝ :=
  m * |(pos_i : ℝ) - (pos_j : ℝ)|

def w2_score_alibi (t1 t2 : GaussianToken n) (pos_i pos_j : ℕ) (m : ℝ) : ℝ :=
  (w2_score t1 t2) - (alibi_penalty pos_i pos_j m)

/--
Theorem: ALiBi W2 Attention is equivalent to Regularized Optimal Transport cost.
If we define the generalized cost C(t1, t2, i, j) = 1/2 W2^2(t1,t2) + m|i-j|,
then the score is exactly -C.
-/
theorem alibi_is_regularized_cost (t1 t2 : GaussianToken n) (pos_i pos_j : ℕ) (m : ℝ) :
    w2_score_alibi t1 t2 pos_i pos_j m = - (0.5 * w2_dist_sq t1 t2 + m * |(pos_i : ℝ) - (pos_j : ℝ)|) := by
  unfold w2_score_alibi w2_score alibi_penalty
  ring

/--
Theorem: ALiBi does not affect the covariance structure.
Just as RoPE only affects means, ALiBi is an additive term that is independent of the Gaussian parameters.
It acts as a prior on the positions.
-/
theorem alibi_independent_of_sigma (t1 t2 : GaussianToken n) (pos_i pos_j : ℕ) (m : ℝ) :
    w2_score_alibi t1 t2 pos_i pos_j m = -0.5 * (LinearAlgebra.dist t1.mu t2.mu ^ 2 + diag_trace_term t1 t2) - m * |(pos_i : ℝ) - (pos_j : ℝ)| := by
  unfold w2_score_alibi w2_score w2_dist_sq alibi_penalty
  rfl

end W2Attn
