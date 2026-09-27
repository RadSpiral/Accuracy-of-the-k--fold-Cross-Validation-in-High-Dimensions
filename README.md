# Simulation code for *Accuracy of the k fold Cross Validation in High Dimensions*

This script reproduces the neural network experiment in Section 3.2 (Figure 1) of the paper. It asks whether the rates proved for strongly convex estimators also describe k fold CV for a network trained by SGD, a setting outside the theory.

## What it tests

For the estimators covered by the paper, Theorems 2.1 and 2.2 give

    |E[CV(k) − OO]| = Õ(1/k),    Var(CV(k) − OO) = Õ(1/n),

so the mean squared error of CV is Õ(k⁻² + n⁻¹). With n ∝ p and k ≍ n^a, a regression of 2 log|CV(k) − OO| on log p should have slope at most −min(2a, 1), and close to −2a when the training size bias dominates. The script compares four rules, k = 4, ⌈n^0.25⌉, ⌈n^0.5⌉ and ⌈n^0.75⌉, for which these slopes are 0, −0.5, −1 and −1 (or −1.5 when bias dominates).

## Setup

A teacher network x ↦ w₂ᵀ ReLU(W₁ᵀ x) with h = 5 hidden units and no biases generates y with Gaussian noise of standard deviation 0.1, where x ~ N(0, I_d). The fitted network has the same architecture, so p = 5(d + 1). We take d = 100, 200, 400, 800, giving p = 505 to 4005, and n = ⌊p/2⌋, so γ = p/n ≈ 2 in the paper's notation. Note that the variable `gamma` in the code is n/p = 0.5.

Every fit starts from a fresh random initialization and runs minibatch SGD (batch 16, step size 0.05) for the same budget of `total_steps` steps, with no explicit penalty. The fixed step budget plays the role of the fixed λ in the paper, so OO is the risk of the full data fit under this fitting rule. OO is estimated on 10n fresh test points, and each setting is repeated 50 times.

## Running

Requires `numpy`, `torch`, `scikit-learn`, `scipy` and `matplotlib`. All settings are in the CONFIG block at the top of the script.

    python kcv_mlp_simulation.py

Results are checkpointed to `./results/checkpoint.pkl` after every replicate, and an interrupted run resumes where it stopped. The checkpoint does not record the configuration, so delete `./results` before rerunning with new settings. The script writes `config.txt` and one figure per rule (`rate_K4.png`, `rate_Kn025.png`, `rate_Kn05.png`, `rate_Kn075.png`), which are the four panels of Figure 1.

## Citation

Please cite the paper if you use this code. Citation details will be added on publication.
