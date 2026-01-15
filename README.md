# High-Dimensional Neural Network Potential for the Malonaldehyde Molecule  
**Brandon-Lee De Bruyn**

---

## Introduction

This project implements a **second-generation high-dimensional neural network potential (HDNNP)** as proposed by Behler–Parrinello, where molecular energies are represented as sums of element-specific atomic contributions predicted from local, symmetry-preserving descriptors [1,2]. A structural overview of the second-generation HDNNP pipeline is shown below.

<p align="center">
  <img src="HDNNP_Structure.png" width="420">
</p>

**Figure 1.** Second-generation HDNNP structure: Cartesian coordinates  
$\mathbf{R}_i^\mu \rightarrow$ ACSF descriptors $\mathbf{G}_i^\mu \rightarrow$ element-specific atomic NNs $\rightarrow$ atomic energies summed to total energy [1,2].

---

## Data Curation

For more efficient training, data splitting was performed at the **frame (molecule) level**. Unique frames were partitioned into training and validation sets using a **90/10 split** (`val_size = 0.1`, `random_state = 42`), resulting in a train/validation/test frame distribution of:

- **Training:** 16,200  
- **Validation:** 1,800  
- **Test:** 3,600  

---

## Atom-Centered Symmetry Functions (ACSFs)

Since the HDNNP requires descriptors invariant to translation, rotation, and permutation of identical atoms, each atomic environment was encoded using **radial and angular ACSFs** [1,2]. All symmetry functions were localized using the smoothly decaying cosine cutoff function [2]:

\[
f_c(R_{ij}) =
\begin{cases}
\frac{1}{2}\left[\cos\left(\pi \frac{R_{ij}}{R_{\mathrm{cut}}}\right)+1\right], & R_{ij} \le R_{\mathrm{cut}}, \\
0, & R_{ij} > R_{\mathrm{cut}},
\end{cases}
\]

with $R_{\mathrm{cut}} = 5.5$ and $R_{ij} = \lVert \mathbf{R}_j - \mathbf{R}_i \rVert$.

---

### Element-Resolved Radial Functions

Radial ACSFs describe the neighbor distribution around atom $i$. Element-resolved radial functions were used [2]:

\[
G^{\mathrm{rad}}_{i,m} =
\sum_{\substack{j \ne i \\ R_{ij} \le R_{\mathrm{cut}}}}
\exp\!\left[-\eta_m (R_{ij}-R_s)^2\right]\, f_c(R_{ij}).
\]

Parameter grid:

- $\eta \in \{0.05,\;0.5,\;1.0,\;2.0,\;4.0,\;8.0\}$
- $R_s \in \text{linspace}(0, R_{\mathrm{cut}}, 8)$

This yields $6 \times 8 = 48$ radial functions **per neighbor element type**  
$Z \in \{\mathrm{C}, \mathrm{H}, \mathrm{O}\}$, giving:

\[
3 \times 48 = 144 \quad \text{radial features per atom.}
\]

---

### Element-Pair-Resolved Angular Functions

Radial functions alone cannot distinguish environments with identical neighbor radii but different angular arrangements. Therefore, angular ACSFs were also computed [2]:

\[
\begin{aligned}
G^{\mathrm{ang}}_{i,m}
&=
2^{1-\zeta_m}
\sum_{\substack{j<k \\ R_{ij},R_{ik}\le R_{\mathrm{cut}}}}
\left(1+\lambda_m \cos\theta_{ijk}\right)^{\zeta_m} \\
&\quad \times
\exp\!\left[-\eta_m \left(R_{ij}^2+R_{ik}^2+R_{jk}^2\right)\right]
f_c(R_{ij})f_c(R_{ik}),
\end{aligned}
\]

where $\theta_{ijk}$ is the angle at the central atom $i$.

Parameter grid:

- $\eta \in \{0.0005,\;0.005\}$
- $\zeta \in \{1,\;2,\;4\}$
- $\lambda \in \{-1,\;1\}$

This yields $2 \times 3 \times 2 = 12$ angular functions per **unordered element pair**.  
With three elements:

\[
(\mathrm{C,C}), (\mathrm{C,H}), (\mathrm{C,O}), (\mathrm{H,H}), (\mathrm{H,O}), (\mathrm{O,O})
\]

giving:

\[
6 \times 12 = 72 \quad \text{angular features per atom.}
\]

---

### Final Descriptor Dimension

\[
D = 144 + 72 = 216
\]

---

## Model Design and Training

The model follows the **second-generation HDNNP architecture** shown in Figure 1. Each atomic network maps the local ACSF descriptor of an atom to a scalar atomic energy contribution, and the total molecular energy is obtained by summation:

\[
\hat{E} = \sum_{i=1}^{N} \hat{E}_i
= \sum_{\mu}\sum_{i \in \mu} \mathrm{NN}_{\mu}(\mathbf{G}_i^\mu).
\]

For malonaldehyde, **three element-specific atomic neural networks** (C, H, O) were constructed. Each network used:

- Input dimension: $D = 216$
- Two fully connected hidden layers (64 neurons each)
- $\tanh$ activation
- Linear output layer

### Regularization

To reduce overfitting:

- **L2 weight regularization** with $\lambda = 10^{-6}$ on all dense layers
- **Dropout** with rate $0.05$ after each hidden layer [3]

---

### Normalization and Optimization

Feature and energy standardization:

\[
X' = \frac{X - \mu_X}{\sigma_X}, \qquad
y' = \frac{y - \mu_y}{\sigma_y}.
\]

Training details:

- Loss: Mean-squared error (MSE)
- Optimizer: Adam [4]
- Learning rate: $10^{-4}$
- Batch size: 32
- Epochs: up to 500
- Callbacks:
  - Early stopping (patience = 30)
  - Reduce-on-plateau (factor = 0.25, min\_lr = $10^{-6}$)

<p align="center">
  <img src="loss_curve.png" width="500">
</p>

**Figure 2.** Training and validation MSE versus epoch.

---

## Testing and Results

Metrics:

\[
\mathrm{MAE} = \frac{1}{F}\sum_f |\hat{y}_f - y_f|, \qquad
\mathrm{RMSE} = \sqrt{\frac{1}{F}\sum_f (\hat{y}_f - y_f)^2}
\]

Final test performance:

- **MAE:** 0.737  
- **RMSE:** 0.996  
- **$R^2$:** 0.721  

<p align="center">
  <img src="pred_vs_true.png" width="480"><br>
  <img src="resid_vs_true.png" width="480">
</p>

**Figure 3.**  
(Top) Predicted vs. true test energies.  
(Bottom) Residuals (true − predicted) vs. true energy.

The results show strong agreement between predicted and reference energies, with residuals broadly centered around zero, indicating that the ACSF-based second-generation HDNNP captures the dominant structure of the malonaldehyde energy landscape.

---

## References

1. Behler, J., & Parrinello, M. *Generalized neural-network representation of high-dimensional potential-energy surfaces*. **Phys. Rev. Lett.**, 98, 146401 (2007).  
2. Behler, J. *Four generations of high-dimensional neural network potentials*. **Chem. Rev.**, 121, 10037–10072 (2021).  
3. Hinton, G. et al. *Improving neural networks by preventing co-adaptation of feature detectors* (2012).  
4. Kingma, D. P., & Ba, J. *Adam: A method for stochastic optimization* (2014).
