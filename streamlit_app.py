import pandas as pd
import streamlit as st
import joblib
from sklearn.linear_model import LassoCV
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="AstroLasso model regressors")

def plot_coefficients_lasso(ref_spectrum: np.array, label_name: str, model,
                            wavelengths_array: np.array, moore_rays: pd.DataFrame):
    """
    Plots the coefficients of the considered lasso model using Plotly graph_objects.

    :param ref_spectrum: array to be plotted as a line with a secondary y-axis
    :param label_name: name of the label
    :param model: LassoCV model considered
    :param wavelengths_array: numpy array of wavelengths
    :param trained_on_synth: boolean value is True if the model is trained on synthetic spectra
    :param binning: binning size (positive float) - zero if no binning is applied
    :return: a Plotly graph object with the coefficients position and importance
    """

    coeff_dict = {
        float(wavelength): float(coeff)
        for wavelength, coeff in zip(wavelengths_array, model.coef_)
        if abs(coeff) > 1e-4
    }

    fig = go.Figure()

    for wavelength, coeff in coeff_dict.items():
        fig.add_trace(
            go.Scatter(
                x=[wavelength, wavelength],
                y=[0, coeff],
                mode="lines",
                line=dict(color='black', width=1),
                showlegend=False
            )
        )

    fig.add_trace(
        go.Scatter(
            x=list(coeff_dict.keys()),
            y=list(coeff_dict.values()),
            mode='markers',
            marker=dict(color='black', size=5),
            name='Regressor'
        )
    )

    for row in moore_rays.itertuples():
        fig.add_trace(
            go.Scatter(
                x=[row.wavelength, row.wavelength],
                y=[0, 1.1 * max(coeff_dict.values())],
                mode='lines',
                line=dict(color='green', width=0.5),
                name='Moore ray',
                opacity=0.2,
                hovertemplate=f"{row.element} : {row.wavelength} \u00C5",
            ),
        )

    fig.add_trace(
        go.Scatter(
            x=wavelengths_array.tolist(),
            y=ref_spectrum.tolist(),
            mode="lines",
            line=dict(color="red", width=0.5, dash="solid"),
            opacity=0.5,
            name="Reference spectrum",
            yaxis="y2"
        )
    )

    fig.update_layout(
        title=f"Regressors for {label_name} model",
        xaxis_title="Wavelength [\u00C5]",
        yaxis_title=f"Regressors",
        yaxis2=dict(
            title="Reference Spectrum",
            overlaying="y",
            side="right",
            showgrid=False,
            titlefont=dict(color="red"),
        ),
        template="plotly_white",
        width=800,
        height=400,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True)
    )

    return fig

w = np.load('data/uves_wavelength.npz')['wavelength']
obs_models = joblib.load('models/lasso_models_dict_full_definition_obs.joblib')
synth_models = joblib.load('models/lasso_models_dict_full_definition_synth.joblib')
solar_spectrum = np.load("data/solar_spectrum.npy")
synth_ref_spectrum = np.load("data/synth_ref_spectrum.npy")
moore_rays = pd.read_csv("data/ll_moore.dat", sep='\s+', names=['element', 'wavelength'])
moore_rays = moore_rays[(moore_rays.wavelength >= min(w)) & (moore_rays.wavelength <= max(w))]

if __name__ == "__main__":
    labels = ["Models trained on OBSERVED spectra", "Models trained on SYNTHETIC spectra"]
    tab1, tab2 = st.tabs(labels)
    with tab1:
        for lab in obs_models.keys():
            if isinstance(obs_models[lab], LassoCV):
                fig = plot_coefficients_lasso(solar_spectrum, lab, obs_models[lab], w, moore_rays)
                st.plotly_chart(fig, use_container_width=True)
    with tab2:
        for lab in synth_models.keys():
            if isinstance(synth_models[lab], LassoCV):
                fig = plot_coefficients_lasso(synth_ref_spectrum, lab, synth_models[lab], w, moore_rays)
                st.plotly_chart(fig, use_container_width=True)
