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

number_of_samples = 10 ** col1.slider("Num. of Samples: Order of Magnitude", 1, 4, 2)
true_estimator_std = col2.slider("Estimator Standard Devietion", 1, 50, 10)
measurment_std = col3.slider("Measurments Standard Devietion", 1, 10, 5)

# generate samples
with st.spinner("Wait for it..."):
    estimtor_points = random.normal(
        loc=0.0, scale=true_estimator_std, size=[number_of_samples, 2]
    )
    measurment_points = random.normal(
        loc=0.0, scale=measurment_std, size=[number_of_samples, 2]
    )
st.success("Simulation is done!")

difference_vectors = estimtor_points - measurment_points

# plot
hist_data = [estimtor_points, measurment_points, difference_vectors]
hist_data_norm = [np.linalg.norm(points, axis=1) for points in hist_data]


group_labels = ["Estimators", "Measurments", "Differences"]

fig = ff.create_distplot(hist_data_norm, group_labels)  # create plotly object

st.plotly_chart(fig, use_container_width=True)
