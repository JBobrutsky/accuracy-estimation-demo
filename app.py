import streamlit as st
import numpy as np
from numpy import random
import plotly.figure_factory as ff


st.title("Accuracy Estimation")

st.write("Accuracy estimation using noisy sampling for geographical applications.")

st.latex(
    r"""X \sim \mathcal{N}(\mu, P) = \mathcal{N}(\mu, \overbrace{\theta P_0}^{P})\\
    Y \sim \mathcal{N}(\mu, G)\\
    Z = X-Y \sim \mathcal{N}(0, P+G)"""
)

# parmeters
col1, col2, col3 = st.columns(3)

number_of_samples = 10 ** col1.slider("Num. of Samples: Order of Magnitude", 1, 5, 2)
true_estimator_std = col2.slider("Estimator Standard Devietion", 1, 50, 10)
measurment_std = col3.slider("Measurments Standard Devietion", 1, 10, 5)

# generate samples
def sample(number_of_samples=100, std=1.0):
    return random.normal(loc=0.0, scale=std, size=[number_of_samples, 2])


estimtor_points = sample(number_of_samples, true_estimator_std)
measurment_points = sample(number_of_samples, measurment_std)

difference_vectors = estimtor_points - measurment_points

# plot
hist_data = [estimtor_points, measurment_points, difference_vectors]
hist_data_norm = [np.linalg.norm(points, axis=1) for points in hist_data]

group_labels = ["Estimators", "Measurments", "Differences"]

with st.spinner("Wait for it..."):
    fig = ff.create_distplot(hist_data_norm, group_labels)  # creates plotly object
    st.plotly_chart(fig, use_container_width=True)

st.header("Maximum Liklihood Estimator")

st.markdown(
    r"""
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
"""
)
