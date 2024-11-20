import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression


def logistic_regression(
    intensities: pd.DataFrame | np.ndarray, weights: pd.DataFrame, plot: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode intensities using logistic regression.

    Parameters
    ----------
    intensities
        A (N,C) array of intensities where N is the number of spots and C is the number of channels.
    weights
        A dataframe with weights for the logistic regression model.
    plot
        Whether to plot heatmap of intensities annotated with decoded values.

    Returns
    -------
    decoding
        a DataFrame of spots with decoded value, prob, intensity, and snr columns
    bit_performance
        a DataFrame with bit, intensity, and snr columns
    """
    # Setup
    index = None
    bias = weights.bias.values
    weights = weights.drop(columns="bias")
    if isinstance(intensities, pd.DataFrame):
        index = intensities.index
        intensities = intensities.loc[:, weights.columns].values
    classifier = LogisticRegression()
    classifier.coef_ = weights.values
    classifier.intercept_ = bias
    classifier.classes_ = weights.index
    # Normalization
    normalized = intensities + 1e-6
    normalized /= np.sum(normalized, axis=1, keepdims=True)
    # Prediction
    probs = classifier.predict_proba(normalized)
    max_prob = np.max(probs, axis=1)
    max_index = np.argmax(probs, axis=1)
    value = weights.index[max_index]
    # Get intensity and snr
    signal_mask = (weights == weights.max(axis=0)).values[max_index, :]
    signal_intensity = np.ma.masked_array(intensities, ~signal_mask)
    noise_intensity = np.ma.masked_array(intensities, signal_mask)
    spot_intensity = np.ma.mean(signal_intensity, axis=1)
    spot_snr = spot_intensity / (np.ma.mean(noise_intensity, axis=1) + 1e-6)
    bit_intensity = np.ma.mean(signal_intensity[max_prob > 0.5], axis=0)
    bit_snr = bit_intensity / (np.ma.mean(noise_intensity[max_prob > 0.5], axis=0) + 1e-6)
    # Format results
    decoding = pd.DataFrame(
        {"value": value, "prob": max_prob, "intensity": spot_intensity, "snr": spot_snr}, index=index
    )
    bit_performance = pd.DataFrame({"bit": weights.columns, "intensity": bit_intensity, "snr": bit_snr})
    # Plot
    if plot:
        normalized = pd.DataFrame(
            normalized[decoding.prob > 0.5], index=decoding[decoding.prob > 0.5].value.values, columns=weights.columns
        )
        if len(normalized) > 1000:
            normalized = normalized.sample(1000)
        sns.heatmap(normalized.sort_index(), vmax=1)
    return decoding, bit_performance
