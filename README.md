# Accuracy Estimation Demo App

Accuracy estimation using noisy sampling for geographical applications.

$$X \sim \mathcal{N}(\mu, P) = \mathcal{N}(\mu, \overbrace{\theta P_0}^{P})\\
    Y \sim \mathcal{N}(\mu, G)\\
    Z = X-Y \sim \mathcal{N}(0, P+G)$$

## Maximum Liklihood Estimator
$$
\mathcal{L}(\theta|z) = \frac{1}{2\pi\sqrt{\det(D)} }\exp{(-\frac{1}{2} z^T D^{-1} z)}
$$

when $D = P+G = \theta P_0 + G$.

Allowing combining multilpe measurmetns with different variances:
$$
        \mathcal{L}(\theta|z_1, z_2, ..., z_n) = \prod_{i=1}^{n}{\frac{1}{2\pi\sqrt{\det(D_i)} }\exp{(-\frac{1}{2} z_i^T D_i^{-1} z_i)}}
$$
The log-liklihood would be:
$$
\ell(\theta|\{z_i\}_{i=1}^n) =   -n\log{2\pi} - \frac{1}{2}\sum_{i=1}^{n}{\log{(\det{D_i})}}  - \frac{1}{2}\sum_{i=1}^{n}{z_i^T D_i^{-1} z_i}
$$

Such that the MLE estimator is:
$$
\hat\theta = \argmin_{\theta}{\sum_{i=1}^{n}{\log{(\det{D_i})}  + z_i^T D_i^{-1} z_i}}
$$

In order to efficenly solve, we can calculate the gradient:
$$
-\nabla\ell(\theta|\{z_i\}_{i=1}^n) = \sum_{i=1}^{n}Tr(D_i^{-1}\frac{d D_i}{d\theta}) - z_i^T D_i^{-1} \frac{d D_i}{d\theta} D_i z_i
$$
and using $\frac{d D_i}{d\theta} = P_i$:
$$
-\nabla\ell(\theta|\{z_i\}_{i=1}^n) = \sum_{i=1}^{n}Tr(D_i^{-1}P_i) - z_i^T D_i^{-1} P_i D_i z_i
$$
