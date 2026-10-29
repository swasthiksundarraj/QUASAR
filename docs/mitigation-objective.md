# Tiny-instance QUBO mitigation objective

The current optimizer chooses exactly one scoped mitigation candidate or the explicit no-op option for each incident. For action variable \(x_i \in \{0,1\}\), scenario risk \(r_s \in [0,1]\), and covered scenarios \(S_i\), the utility is:

\[
u_i = b_i \sum_{s \in S_i} r_s - a_i - d_i - c_i
      - \lambda_k \frac{k_i}{\max(K,1)}
\]

where \(b_i\) is security benefit, \(a_i\) availability cost, \(d_i\) service disruption, \(c_i\) mitigation cost, \(k_i\) required key bits, and \(K\) available key capacity. The no-op has zero utility. The minimized QUBO is:

\[
E(x) = -\sum_i u_i x_i + P\left(\sum_i x_i - 1\right)^2
       + P\sum_i I_i x_i
\]

where \(P\) is the configurable penalty and \(I_i=1\) if an action alone exceeds key capacity or the configured service-disruption limit. This expansion uses diagonal terms \(-u_i-P+PI_i\), pair terms \(2P x_i x_j\), and constant \(P\). The independent verifier remains authoritative: it checks one-hot encoding, exact scope allowlists, action types, capacity, and service impact before the mock adapter is invoked. A QAOA sample that violates any check is rejected in favor of no-op.

The prototype intentionally limits a plan to one action per incident rather than approximating a multi-action knapsack. The exact enumerator is the comparison oracle for tiny instances. QAOA is approximate; solver runtime, sampled objective, expected objective, and feasibility are returned separately. No quantum advantage is claimed.

