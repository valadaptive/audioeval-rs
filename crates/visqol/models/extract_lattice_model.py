"""Extract lattice_speech_model.bin from the upstream ViSQOL TFLite model.

Usage:
    pip install tflite numpy
    python3 extract_lattice_model.py path/to/lattice_..._raw.tflite

The upstream model is a TensorFlow Lattice "calibrated lattice ensemble"
compiled to ordinary TFLite arithmetic ops:

  1. one piecewise-linear calibrator per scalar input (fvnsim{0..20},
     fvnsim10_{0..20}, fstdnsim{0..20}, fvdegenergy{0..20}, tau), each
     producing one calibrated value per lattice that consumes the feature;
     fvnsim/fvnsim10 calibrators carry a missing-value branch selected by
     exact f32 equality against a sentinel,
  2. 60 rank-12 lattices with 2 vertices per dimension: multilinear
     interpolation over 2^12 corner values, with per-dimension weights
     clipped as 1 - min(|v - vertex|, 1),
  3. a linear combination of the 60 lattice outputs,
  4. one final piecewise-linear output calibrator.

This script contains a minimal interpreter for the 16 builtin TFLite ops the
model uses (needed to record runtime tensor shapes), traces the graph
structure with assertions at every step, and writes the parameters to a
compact little-endian binary. See src/lattice.rs for the format definition
and the matching inference implementation.
"""
import struct
import sys

import numpy as np
import tflite

FEATURE_NAMES = (
    [f'fvnsim{i}' for i in range(21)]
    + [f'fvnsim10_{i}' for i in range(21)]
    + [f'fstdnsim{i}' for i in range(21)]
    + [f'fvdegenergy{i}' for i in range(21)]
    + ['tau']
)

TYPES = {
    tflite.TensorType.FLOAT32: np.float32,
    tflite.TensorType.INT32: np.int32,
    tflite.TensorType.BOOL: np.bool_,
}


class Model:
    """Flatbuffer wrapper plus a minimal interpreter for the ops used."""

    def __init__(self, path):
        self.buf = open(path, 'rb').read()
        self.m = tflite.Model.GetRootAsModel(self.buf, 0)
        self.g = self.m.Subgraphs(0)
        self.opname = []
        for i in range(self.m.OperatorCodesLength()):
            oc = self.m.OperatorCodes(i)
            code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
            self.opname.append(
                [k for k, v in vars(tflite.BuiltinOperator).items()
                 if v == code][0])
        s = self.m.SignatureDefs(0)
        self.sig_in = {s.Inputs(i).Name().decode(): s.Inputs(i).TensorIndex()
                       for i in range(s.InputsLength())}
        self.sig_out = {s.Outputs(i).Name().decode(): s.Outputs(i).TensorIndex()
                        for i in range(s.OutputsLength())}

    def const(self, ti):
        t = self.g.Tensors(ti)
        b = self.m.Buffers(t.Buffer())
        if b.DataLength() == 0:
            return None
        arr = np.frombuffer(b.DataAsNumpy().tobytes(), dtype=TYPES[t.Type()])
        shape = [t.Shape(j) for j in range(t.ShapeLength())]
        return arr.reshape(shape) if shape else arr.reshape(())

    def run(self, feed):
        """feed: signature-input-name -> float. Returns all tensor values."""
        g = self.g
        vals = {}
        for name, ti in self.sig_in.items():
            vals[ti] = np.array([np.float32(feed[name])], dtype=np.float32)
        for i in range(g.TensorsLength()):
            c = self.const(i)
            if c is not None:
                vals[i] = c

        for oi in range(g.OperatorsLength()):
            op = g.Operators(oi)
            name = self.opname[op.OpcodeIndex()]
            outs = [op.Outputs(j) for j in range(op.OutputsLength())]
            x = [vals[op.Inputs(j)] if op.Inputs(j) != -1 else None
                 for j in range(op.InputsLength())]

            if name == 'EXPAND_DIMS':
                r = np.expand_dims(x[0], int(x[1]))
            elif name == 'ABS':
                r = np.abs(x[0])
            elif name == 'RELU':
                r = np.maximum(x[0], 0)
            elif name == 'MINIMUM':
                r = np.minimum(x[0], x[1])
            elif name == 'MUL':
                r = x[0] * x[1]
            elif name == 'ADD':
                r = x[0] + x[1]
            elif name == 'SUB':
                r = x[0] - x[1]
            elif name == 'EQUAL':
                r = x[0] == x[1]
            elif name == 'CAST':
                r = x[0].astype(TYPES[g.Tensors(outs[0]).Type()])
            elif name == 'SHAPE':
                r = np.array(x[0].shape, dtype=np.int32)
            elif name == 'FILL':
                r = np.full(tuple(x[0]), x[1], dtype=x[1].dtype)
            elif name == 'RESHAPE':
                r = x[0].reshape(tuple(int(v) for v in x[1]))
            elif name == 'CONCATENATION':
                o = tflite.ConcatenationOptions()
                o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
                assert o.FusedActivationFunction() == 0
                r = np.concatenate(x, axis=o.Axis())
            elif name == 'SPLIT':
                o = tflite.SplitOptions()
                o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
                parts = np.split(x[1], o.NumSplits(), axis=int(x[0]))
                for t, p in zip(outs, parts):
                    vals[t] = p
                continue
            elif name == 'FULLY_CONNECTED':
                o = tflite.FullyConnectedOptions()
                o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
                assert o.FusedActivationFunction() == 0
                assert o.WeightsFormat() == 0
                r = x[0] @ x[1].T
                if len(x) > 2 and x[2] is not None:
                    r = r + x[2]
            elif name == 'BATCH_MATMUL':
                o = tflite.BatchMatMulOptions()
                o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
                a, b = x[0], x[1]
                if o.AdjX():
                    a = a.swapaxes(-1, -2)
                if o.AdjY():
                    b = b.swapaxes(-1, -2)
                r = a @ b
            else:
                raise NotImplementedError(name)
            vals[outs[0]] = r
        return vals


def extract(mdl):
    g = mdl.g

    producer = {}
    consumers = {}
    for oi in range(g.OperatorsLength()):
        op = g.Operators(oi)
        for j in range(op.OutputsLength()):
            producer[op.Outputs(j)] = op
        for j in range(op.InputsLength()):
            consumers.setdefault(op.Inputs(j), []).append(op)

    # Record runtime shapes (declared shapes are unreliable for dynamic dims).
    feed = {name: 0.5 for name in FEATURE_NAMES}
    shape = {t: v.shape for t, v in mdl.run(feed).items()}

    const = mdl.const

    def scalar(t):
        c = mdl.const(t)
        assert c.size == 1
        return float(c.reshape(-1)[0])

    def opname(op):
        return mdl.opname[op.OpcodeIndex()]

    def only_consumer(t, kind=None):
        c = consumers[t]
        assert len(c) == 1, (t, [opname(o) for o in c])
        if kind is not None:
            assert opname(c[0]) == kind, (opname(c[0]), kind)
        return c[0]

    # ---- 1. Calibrators -------------------------------------------------
    # Pattern:
    #   e = EXPAND_DIMS(x)
    #   w = RELU(MINIMUM(MUL(SUB(e, keypoints), inv_lengths), 1.0))
    #   cal = FC(CONCAT(FILL(SHAPE(e), 1.0), w), kernel)       # no bias
    # optionally wrapped in missing-value selection:
    #   m = CAST(EQUAL(e, missing_input_value))
    #   out = ADD(MUL(m, missing_vals), MUL(SUB(1.0, m), cal))
    # then SPLIT into n_units scalars, each feeding one lattice slot.
    calibrators = []
    split_unit = {}   # tensor index -> (feature_idx, unit_idx)
    for fi, fname in enumerate(FEATURE_NAMES):
        expand = only_consumer(mdl.sig_in[fname], 'EXPAND_DIMS')
        e_out = expand.Outputs(0)

        eq = [op for op in consumers[e_out] if opname(op) == 'EQUAL']
        sub_kp = [op for op in consumers[e_out] if opname(op) == 'SUB']
        assert len(sub_kp) == 1 and len(eq) <= 1
        sub_kp = sub_kp[0]
        keypoints = const(sub_kp.Inputs(1))
        mul_len = only_consumer(sub_kp.Outputs(0), 'MUL')
        inv_lengths = const(mul_len.Inputs(1))
        minimum = only_consumer(mul_len.Outputs(0), 'MINIMUM')
        assert scalar(minimum.Inputs(1)) == 1.0
        relu = only_consumer(minimum.Outputs(0), 'RELU')
        concat = only_consumer(relu.Outputs(0), 'CONCATENATION')
        ones = producer[concat.Inputs(0)]
        assert opname(ones) == 'FILL' and scalar(ones.Inputs(1)) == 1.0
        assert concat.Inputs(1) == relu.Outputs(0)
        fc = only_consumer(concat.Outputs(0), 'FULLY_CONNECTED')
        kernel = const(fc.Inputs(1))
        assert fc.InputsLength() == 2 or fc.Inputs(2) == -1  # no bias
        n_units = kernel.shape[0]
        assert kernel.shape == (n_units, keypoints.shape[0] + 1)

        missing_input_value = None
        missing_vals = None
        if eq:
            missing_input_value = scalar(eq[0].Inputs(1))
            m_out = only_consumer(eq[0].Outputs(0), 'CAST').Outputs(0)
            muls = [op for op in consumers[m_out] if opname(op) == 'MUL']
            subs = [op for op in consumers[m_out] if opname(op) == 'SUB']
            assert len(muls) == 1 and len(subs) == 1
            missing_vals = const(muls[0].Inputs(1))
            assert missing_vals.shape == (1, n_units)
            missing_vals = missing_vals[0]
            assert scalar(subs[0].Inputs(0)) == 1.0
            mul_cal = only_consumer(subs[0].Outputs(0), 'MUL')
            assert mul_cal.Inputs(1) == fc.Outputs(0)
            add = only_consumer(mul_cal.Outputs(0), 'ADD')
            assert add.Inputs(0) == muls[0].Outputs(0)
            final = add.Outputs(0)
        else:
            final = fc.Outputs(0)

        split = only_consumer(final, 'SPLIT')
        assert int(scalar(split.Inputs(0))) == 1
        assert split.OutputsLength() == n_units
        for u in range(n_units):
            split_unit[split.Outputs(u)] = (fi, u)

        calibrators.append(dict(
            name=fname, n_units=n_units,
            keypoints=keypoints.astype(np.float32),
            inv_lengths=inv_lengths.astype(np.float32),
            kernel=kernel.astype(np.float32),
            missing_input_value=missing_input_value,
            missing_vals=missing_vals))

    # ---- 2. Lattices ----------------------------------------------------
    # Leaf per lattice input v: SUB(1, MINIMUM(ABS(SUB(v, [0,1])), 1)),
    # the interpolation weight pair [1-min(|v|,1), 1-min(|v-1|,1)].
    # Pairs fold into a 2^12-corner weight vector by repeatedly appending a
    # dimension as the least-significant index: MUL/BATCH_MATMUL of a column
    # (..., N, 1) with a row (..., 1, 2), flattened by RESHAPE.
    def trace_dims(t):
        """Ordered leaf list (most significant first) for tensor t."""
        op = producer[t]
        kind = opname(op)
        if kind in ('RESHAPE', 'EXPAND_DIMS'):
            return trace_dims(op.Inputs(0))
        if kind in ('MUL', 'BATCH_MATMUL'):
            a, b = op.Inputs(0), op.Inputs(1)
            assert shape[a][-1] == 1 and shape[a][-2] > 1, (kind, shape[a])
            assert shape[b][-1] == 2 and shape[b][-2] == 1, (kind, shape[b])
            return trace_dims(a) + trace_dims(b)
        if kind == 'SUB':
            assert scalar(op.Inputs(0)) == 1.0
            minimum = producer[op.Inputs(1)]
            assert opname(minimum) == 'MINIMUM'
            assert scalar(minimum.Inputs(1)) == 1.0
            ab = producer[minimum.Inputs(0)]
            assert opname(ab) == 'ABS'
            sub01 = producer[ab.Inputs(0)]
            assert opname(sub01) == 'SUB'
            assert const(sub01.Inputs(1)).reshape(-1).tolist() == [0.0, 1.0]
            return [split_unit[sub01.Inputs(0)]]
        raise AssertionError(kind)

    # Walk back from the output: output calibrator <- linear combination
    # <- concat of the 60 per-lattice corner-value dot products.
    out_fc = producer[list(mdl.sig_out.values())[0]]
    assert opname(out_fc) == 'FULLY_CONNECTED'
    out_concat = producer[out_fc.Inputs(0)]
    assert opname(out_concat) == 'CONCATENATION'
    out_relu = producer[out_concat.Inputs(1)]
    minimum = producer[out_relu.Inputs(0)]
    assert scalar(minimum.Inputs(1)) == 1.0
    mul_len = producer[minimum.Inputs(0)]
    sub_kp = producer[mul_len.Inputs(0)]
    out_calib = dict(
        keypoints=const(sub_kp.Inputs(1)).astype(np.float32),
        inv_lengths=const(mul_len.Inputs(1)).astype(np.float32),
        kernel=const(out_fc.Inputs(1)).astype(np.float32)[0])

    ens_fc = producer[sub_kp.Inputs(0)]
    assert opname(ens_fc) == 'FULLY_CONNECTED'
    ens_weights = const(ens_fc.Inputs(1)).astype(np.float32)
    assert ens_weights.shape == (1, 60)
    assert ens_fc.InputsLength() == 2 or ens_fc.Inputs(2) == -1
    ens_weights = ens_weights[0]
    lat_concat = producer[ens_fc.Inputs(0)]
    assert opname(lat_concat) == 'CONCATENATION'
    assert lat_concat.InputsLength() == 60

    lattices = []
    for li in range(60):
        fc = producer[lat_concat.Inputs(li)]
        assert opname(fc) == 'FULLY_CONNECTED'
        corners = const(fc.Inputs(1)).astype(np.float32)
        assert corners.shape == (1, 4096)
        assert fc.InputsLength() == 2 or fc.Inputs(2) == -1
        wiring = trace_dims(fc.Inputs(0))
        assert len(wiring) == 12
        lattices.append(dict(wiring=wiring, corners=corners[0]))

    # Every calibrated unit feeds exactly one lattice slot.
    used = [w for l in lattices for w in l['wiring']]
    assert len(used) == len(set(used)) == 720
    return calibrators, lattices, ens_weights, out_calib


def write_binary(path, calibrators, lattices, ens_weights, out_calib):
    out = bytearray()
    out += b'VQLM'
    out += struct.pack('<IIII', 1, len(calibrators), len(lattices), 12)
    for c in calibrators:
        n_kp = len(c['keypoints'])
        has_missing = c['missing_input_value'] is not None
        out += struct.pack('<III', c['n_units'], n_kp, int(has_missing))
        out += struct.pack('<f', c['missing_input_value'] or 0.0)
        out += c['keypoints'].tobytes()
        out += c['inv_lengths'].tobytes()
        out += c['kernel'].tobytes()
        if has_missing:
            out += c['missing_vals'].astype(np.float32).tobytes()
    for l in lattices:
        for fi, u in l['wiring']:
            out += struct.pack('<II', fi, u)
    for l in lattices:
        out += l['corners'].tobytes()
    out += np.asarray(ens_weights, np.float32).tobytes()
    out += struct.pack('<I', len(out_calib['keypoints']))
    out += out_calib['keypoints'].tobytes()
    out += out_calib['inv_lengths'].tobytes()
    out += np.asarray(out_calib['kernel'], np.float32).tobytes()
    open(path, 'wb').write(out)
    return len(out)


def direct_predict(calibrators, lattices, ens_weights, out_calib, feed):
    """Vectorized reimplementation used to self-check the extraction."""
    def pwl(x, kp, il, kernel):
        w = np.clip((np.float32(x) - kp) * il, 0.0, 1.0, dtype=np.float32)
        return kernel[:, 0] + kernel[:, 1:] @ w

    cal = []
    for c in calibrators:
        x = np.float32(feed[c['name']])
        if c['missing_input_value'] is not None and \
                x == np.float32(c['missing_input_value']):
            cal.append(c['missing_vals'])
        else:
            cal.append(pwl(x, c['keypoints'], c['inv_lengths'], c['kernel']))
    ens = np.float32(0.0)
    for l, ew in zip(lattices, ens_weights):
        weights = np.ones(1, np.float32)
        for fi, u in l['wiring']:
            v = cal[fi][u]
            pair = 1.0 - np.minimum(np.abs(v - np.array([0.0, 1.0], np.float32)), 1.0)
            weights = np.outer(weights, pair).reshape(-1)
        ens += ew * (l['corners'] @ weights)
    ok = out_calib
    return float(ok['kernel'][0]
                 + ok['kernel'][1:] @ np.clip(
                     (ens - ok['keypoints']) * ok['inv_lengths'], 0.0, 1.0))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        '../../../visqol/model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12'
        '_bs2048_learn.005_ep2400_train1_7_raw.tflite')
    mdl = Model(path)
    parts = extract(mdl)
    n = write_binary('lattice_speech_model.bin', *parts)
    print(f'wrote lattice_speech_model.bin ({n} bytes)')

    # Self-check: direct evaluation of the extracted parameters must agree
    # with interpreting the graph (up to f32 summation-order noise).
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20):
        feed = {}
        for name in FEATURE_NAMES:
            lo, hi = (0.0, 30.0) if name.startswith('fvdegenergy') else (0.0, 1.0)
            feed[name] = float(rng.uniform(lo, hi))
        feed['tau'] = 0.5
        a = direct_predict(*parts, feed)
        b = float(mdl.run(feed)[list(mdl.sig_out.values())[0]].reshape(()))
        worst = max(worst, abs(a - b))
    assert worst < 1e-4, worst
    print(f'self-check passed (max |direct - interpreted| = {worst:.2e})')


if __name__ == '__main__':
    main()
