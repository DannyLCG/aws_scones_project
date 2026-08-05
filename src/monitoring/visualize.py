"""Confidence-over-time and prediction-grid visualizations from captured
Model Monitor data. Ports starter.ipynb cells 89-95.
"""


def plot_confidence_over_time(records: list[dict], threshold: float = 0.8):
    """Scatter inference confidence vs. timestamp, flagging points below `threshold`."""
    raise NotImplementedError


def plot_prediction_grid(records: list[dict], num_images: int = 9):
    """Render a grid of captured input images next to their predicted label."""
    raise NotImplementedError
