# Kernels and Feature Map
The paper that makes me want to dive deep into kernel functions and feature map is [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668).



My tutorial will probably follow the theory of kernel functions and feature maps, kernel SVM(because this is the first time I know this), feature map in this paper, some simulations, why people use kernel functions, some other frontier papers that also use this technique. 

## Theory

### Definitions
- feature: $\mathbf{x_1}, \mathbf{x_2},..., \mathbf{x_n}$, where $\mathbf{x_i} \in \mathbb{R}^{d}$, $\forall 1 \leq i \leq n$
- feature map: map features to features $\phi: \mathbb{R}^{d} \to \mathbb{R}^{D}$, $\mathbf{x} \mapsto \phi(\mathbf{x})$ (typically $D \gg d$, since the goal is to lift the features into a much higher-dimensional space to capture richer feature interactions, especially nonlinear ones.) $\phi(\mathbf{x})$ is called the feature representation(embedding) of $\mathbf{x}$
- kernel (also positive semi-definite kernel): Let \( X \ne \emptyset \) be a set. A function \( k : X \times X \to \mathbb{R} \) is called a positive definite kernel if
  1. Symmetric:  
   \[
   k(x, y) = k(y, x), \quad \forall x, y \in X
   \tag{1.2.1}
   \]

  2. Positive semi-definite: For all \( n \in \mathbb{N} \), for all \( \alpha_1, \ldots, \alpha_n \in \mathbb{R} \) and
   \( x_1, \ldots, x_n \in X \), we have
   \[
   \sum_{i=1}^{n} \sum_{j=1}^{n}
   \alpha_i \alpha_j \, k(x_i, x_j)
   \ge 0
   \tag{1.2.2}
   \]
  This definition is also equivalent to define the kernel matrix
  $$K=
\begin{pmatrix}
k(x_1, x_1) & \cdots & k(x_1, x_n) \\
\vdots      & \ddots & \vdots      \\
k(x_n, x_1) & \cdots & k(x_n, x_n)
\end{pmatrix}
$$
  is positive semi-definite.
  Hence, A kernel function defines the entries of a kernel matrix, i.e., $K_{ij} = k(x_i, x_j)$, and guarantees that such matrices are always PSD for any choice of inputs.
- Reproducing kernel Hilbert spaces (RKHSs): Given a kernel $k$, for any $x \in X$, we obtain a function $k(x, \cdot) : X \to \mathbb{R}$. This family of functions (as \(x\) varies over \(X\)) forms the basic building blocks of the RKHS \(\mathcal{H}_k\)
  \[
\mathcal{H}_k = \Big\{ \text{limits of finite linear combinations of functions of the form } 
k(x, \cdot), \; x \in X \Big\}
\]
  Formally, given a positive semi-definite kernel $k : X \times X \to \mathbb{R}$ on a set $X$, the corresponding reproducint kernel Hilbert Space $\mathcal{H}_k$ is characterised by the following two properties:
  1. For all $x\in X$, we have $k(x, ·) \in \mathcal{H}_k$
  2. The reproducing property holds:
      $$ f(x) = \langle f, k(x, ·)\rangle_{\mathcal{H}_k}, \quad \text{for all } x\in X, f\in \mathcal{H}_k$$


- 
- 



























------
- 
- a kernel is the inner product in the feature space.  
  The kernel function $k$ is defined as $
  k(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x_i})^\top \phi(\mathbf{x}_j)$. The kernel matrix is simply the matrix that stores all pairwise inner products: $
  K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$. By definition, the kernel matrix $K \in \mathbb{R}^{n \times n}$ is symmetric (a naive property)

### Kernel Trick
A kernel trick is where we can kernalize a model(obviously not a random model, surely can work for linear model, riguously requirement for the model see below) to efficiently learn a model over feature representation.

#### General Setting

We have data

$$
(x_i, y_i)_{i=1}^N,\quad x_i \in \mathcal{X},\; y_i \in \mathbb{R}.
$$

We want a function

$$
f : \mathcal{X} \to \mathbb{R}
$$

that minimizes regularized empirical risk:

$$
f^* \in \arg\min_{f \in \mathcal{H}_k}
\left[
\frac{1}{N} \sum_{i=1}^N \ell(f(x_i), y_i)
\;+\;
\lambda \, \|f\|_{\mathcal{H}_k}^2
\right],
\qquad \lambda > 0.
$$

Here:

- \( \ell \) is any loss function (e.g., squared loss, logistic loss, etc.)
- \( \mathcal{H}_k \) is the RKHS induced by kernel \(k\)
- \( \| f \|_{\mathcal{H}_k} \) is the RKHS norm (controls function smoothness)



#### Representer Theorem (Key to Kernel Methods)

Let $f^* $ be a minimizer of the above problem.  
Then $ f^* $ always has the form

$$
f^*(x) = \sum_{i=1}^N \alpha_i \, k(x_i, x)
$$

for some coefficients \( \alpha_1, \dots, \alpha_N \in \mathbb{R} \).

This theorem guarantees that even if the feature space is infinite-dimensional,  
the optimal solution lies in the finite span of training-point kernels.

In other words:  
the model is a weighted sum of kernel functions centered at the training points.



#### Why This Enables the Kernel Trick

Even if

$$
\phi(x) = \text{infinite-dimensional vector},
$$

we never compute \( \phi(x) \) explicitly.

We only need kernel evaluations  

$$
k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle.
$$

Thus, infinite-dimensional learning becomes computationally feasible.



#### Training Interpretation

During (sub)gradient descent in RKHS:

$$
f_t = \sum_{i=1}^N \alpha_{i,t} \, k(x_i, \cdot)
$$

Each update adds a kernel function at one data point,  
so we never leave the span

$$
\text{span}\{k(x_1,\cdot), \dots, k(x_N,\cdot)\}.
$$

This matches the representer theorem.



#### Final Trained Model

After optimization, prediction is:

$$
\hat f(x) = \sum_{i=1}^N \alpha_i \, k(x_i, x)
$$

You only store the coefficients \( \alpha_i \)  
and the support points \( x_i \).


## Feature map in this paper



In this paper, feature maps \( \phi(\cdot) \) are used to approximate the softmax kernel in attention.  
The softmax term

$$
\exp\left(\frac{q_i^\top k_j}{\sqrt{d}}\right)
$$

is approximated using a second-order Taylor series expansion.  
The authors choose a feature map \( \phi : \mathbb{R}^d \rightarrow \mathbb{R}^{d^2} \) such that

$
\phi(q_i)^\top \phi(k_j)
= 1+ q_i^\top k_j+ \frac{(q_i^\top k_j)^2}{2}.
$

This converts attention into a linear dot-product form, enabling **linear-time attention** and allowing the model to maintain a **recurrent state** for memory.

To balance **recall capacity vs. memory cost**, queries and keys are projected to a lower dimension \( d' \):

$$
W_q, W_k \in \mathbb{R}^{d \times d'}, \qquad d' \ll d.
$$

This modulates the recurrent state size and controls the memory–recall tradeoff.

The authors evaluate multiple feature maps  
(\( \phi_{\mathrm{ReLU}}, \phi_{\mathrm{PosELU}}, \phi_{\mathrm{Square}}, \phi_{\mathrm{Identity}}, \phi_{\mathrm{CosFormer}}, \phi_{\mathrm{Performer}} \))  
and find the **Taylor feature map** lies on the Pareto frontier: it provides high recall with competitive efficiency and does not increase model parameters.


## Difference between this paper and SVM kernel

Unlike classical kernel methods such as SVMs—where the kernel trick is used to
implicitly operate in a high-dimensional Hilbert space and achieve non-linear decision boundaries—this paper uses feature maps for a different purpose: to approximate the softmax kernel in attention and make attention computation linear in sequence length.

Traditional kernel methods treat \(k(x_i, x_j)\) as a similarity measure and optimize
coefficients \(\alpha_i\) over training samples:

$$
f(x) = \sum_{i=1}^N \alpha_i k(x_i, x),
$$

where the kernel implicitly defines a function class (RKHS).  
In contrast, this paper explicitly chooses a feature map \(\phi(\cdot)\) that approximates
the exponential kernel

$$
\exp(q^\top k),
$$

so that dot products of transformed queries and keys approximate softmax attention:

$$
\phi(q)^\top \phi(k) \approx \exp(q^\top k).
$$

Thus, instead of *learning in an implicit RKHS*, the model uses feature maps to
**linearize attention computation** and enable recurrent memory updates. The goal is
computational efficiency and long-context recall, not margin maximization.


## Modern kernel Ideas

Modern neural architectures continue to leverage kernel ideas, often via explicit feature
maps instead of implicit kernels:

- **Performer** (Choromanski et al., 2020): random Fourier features to approximate softmax
- **CosFormer** (Qin et al., 2022): cosine similarity feature map to build linear attention
- **Hyena / RWKV / Mamba**: implicit long-range convolution kernels instead of explicit attention
- **Neural Tangent Kernel (NTK)** theory: analyzes infinite-width networks as kernels (Jacot et al., 2018)
- **Deep Kernel Learning**: combine neural feature extractors with GP kernels (Wilson et al.)
- **RetNet / RWKV**: recurrent linear kernels for sequence modeling

Broad trend:  
Deep learning now favors **explicit feature maps** and **kernel approximations** to make attention linear, scalable, and recurrent.
