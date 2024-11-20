import streamlit as st
import joblib
from sklearn.linear_model import LassoCV
import numpy as np
import plotly.graph_objects as go


def plot_coefficients_lasso(reference_spectrum: np.array, label_name: str, model,
                            wavelengths_array: np.array, trained_on_synth: bool, binning: float):
    """
    Plots the coefficients of the considered lasso model using Plotly graph_objects.

    :param reference_spectrum: array to be plotted as a line with a secondary y-axis
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
        if coeff > 1e-4
    }

    # Create coefficients dictionary and ensure proper types
    coeff_dict = {
        float(wavelength): float(coeff)
        for wavelength, coeff in zip(wavelengths_array, model.coef_)
        if coeff > 1e-4
    }

    # Create the primary stem plot for coefficients
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(coeff_dict.keys()),
            y=list(coeff_dict.values()),
            mode="lines",
            line=dict(color='black', width=1),
            name='Stem Lines'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(coeff_dict.keys()),
            y=list(coeff_dict.values()),
            mode='markers',
            marker=dict(color='black', size=5),
            name='Markers'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(coeff_dict.keys()),
            y=[0] * len(coeff_dict),
            mode='lines',
            line=dict(color='black', width=0.5),
            name='Baseline'
        )
    )

    fig.update_layout(
        title=f'Stem Plot of Weight {label_name}',
        xaxis_title='Wavelength [Å]',
        yaxis_title=f'Weight {label_name}',
        showlegend=False,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    fig.add_trace(
        go.Scatter(
            x=wavelengths_array.tolist(),  # Convert to a standard Python list
            y=reference_spectrum.tolist(),  # Convert to a standard Python list
            mode="lines",
            line=dict(color="red", width=0.5, dash="solid"),
            opacity=0.5,
            name="Solar spectrum",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Lasso Coefficients for {label_name}",
        xaxis_title="Wavelength [\u00C5]",
        yaxis_title="Weight",
        yaxis=dict(title=f"Weight {label_name}", side="left"),
        yaxis2=dict(
            title="Reference Spectrum",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        template="plotly_white",
        width=800,
        height=400,
    )

    return fig


plot_placeholder_dict = {}
w = np.load('data/uves_wavelength.npz')['wavelength']
obs_models = joblib.load('models/lasso_models_dict_full_definition_obs.joblib')
synth_models = joblib.load('models/lasso_models_dict_full_definition_synth.joblib')
solar_spectrum = np.load("data/solar_spectrum.npy")

if __name__ == "__main__":
    labels = ["Lasso models trained on observed solar_spectrum", "Lasso models trained on synthetic solar_spectrum"]
    tab1, tab2 = st.tabs(labels)
    with tab1:
        for lab in obs_models.keys():
            if isinstance(obs_models[lab], LassoCV):
                plot_placeholder_dict[f"{lab}_observed"] = plot_coefficients_lasso(solar_spectrum[-1], lab, obs_models[lab], w, trained_on_synth=False, binning=0)
    with tab2:
        for lab in synth_models.keys():
            if isinstance(synth_models[lab], LassoCV):
                plot_placeholder_dict[f"{lab}_synthetic"] = plot_coefficients_lasso(solar_spectrum[-1], lab, synth_models[lab], w, trained_on_synth=True, binning=0)