import numpy as np

from evaluation.metrics import evaluate_predictions, expected_calibration_error
from qatnet.circuit import amplitude_features, amplitude_probability_features


def test_metric_shapes_and_calibration():
    truth = np.array([0, 1, 1, 0])
    probabilities = np.array([[.9, .1], [.2, .8], [.4, .6], [.6, .4]])
    result = evaluate_predictions(truth, probabilities.argmax(1), probabilities, ("benign", "attack"))
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
    assert expected_calibration_error(truth, probabilities) > 0


def test_amplitude_features_normalize_and_validate_width():
    values = amplitude_features(np.array([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1)
    try:
        amplitude_features(np.ones((2, 3)))
    except ValueError as exc:
        assert "power of two" in str(exc)
    else:
        raise AssertionError("non-power-of-two width should fail")


def test_amplitude_probability_features_are_normalized_basis_probabilities():
    amplitudes = np.array([
        [3.0, -4.0, 0.0, 0.0],
        [-3.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    probabilities = amplitude_probability_features(amplitudes)

    assert probabilities.shape == (3, 4)
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
    np.testing.assert_allclose(probabilities[0], probabilities[1])
    np.testing.assert_allclose(probabilities[2], [1.0, 0.0, 0.0, 0.0])


def test_amplitude_probability_features_preserve_existing_input_validation():
    for invalid in (np.ones(4), np.ones((2, 3))):
        try:
            amplitude_probability_features(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid amplitude input should fail")


def test_amplitude_encoding_vectors_remain_signed_for_vqc_inputs():
    values = amplitude_features(np.array([[3.0, -4.0, 0.0, 0.0]]))
    np.testing.assert_allclose(values, [[0.6, -0.8, 0.0, 0.0]])
