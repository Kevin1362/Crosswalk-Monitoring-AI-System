
from core.trajectory import TrackHistory


def test_linear_prediction_moves_forward():
    h = TrackHistory(max_points=6)
    h.update(5, 0.0, 0.0, 0.0)
    h.update(5, 1.0, 10.0, 0.0)
    h.update(5, 2.0, 20.0, 0.0)
    estimate = h.estimate(5, horizon_s=1.0)
    assert estimate.vx_px_s > 0
    assert estimate.predicted_x > 20
