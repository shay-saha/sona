"""Age-only local inference; the source's other heads must already be pruned."""

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


class AgeEncoder:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or "age" not in outputs[0].name.lower()
        ):
            raise ValueError(
                "Expected an age-only graph; run fetch_age_encoder.py first"
            )
        self.input = inputs[0].name
        self.output = outputs[0].name

    def predict(self, path):
        signal, rate = sf.read(path, dtype="float32")
        if signal.ndim == 2:
            signal = signal.mean(axis=1)
        if rate != 16000:
            factor = gcd(rate, 16000)
            signal = resample_poly(signal, 16000 // factor, rate // factor).astype(
                np.float32
            )
        signal = signal[-160000:]
        # The published ONNX graph includes its processor; model.yaml has transform: null.
        value = self.session.run([self.output], {self.input: signal[None, :]})[0]
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError("Invalid age output")
        return float(value.ravel()[0]) * 100
