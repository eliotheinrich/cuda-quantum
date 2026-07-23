#!/usr/bin/env python3
# ============================================================================ #
# Expected outcomes for tests:
#   "dem"          -> cudaq.dem_from_kernel yields a valid DEM
#   "nondet"       -> Stim rejects non-deterministic detector
#   "err_compile"  -> compile-time fatal (bad pauli char, wrong type, etc.)
# ============================================================================ #

import cudaq, sys, traceback

nm = cudaq.NoiseModel()   # empty noise model - still tracks noise ops inside kernels

# ============================================================================ #
# CATEGORY 1: All three Paulis and case-insensitivity
# ============================================================================ #

@cudaq.kernel
def x_basic():
    # expected: dem  (errors=1)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def y_basic():
    # expected: dem  (errors=1)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'Y', q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def z_basic():
    # expected: dem  (errors=1)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'Z', q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector(d)

@cudaq.kernel
def x_lowercase():
    # expected: dem  (errors=0)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'x', q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def y_lowercase():
    # expected: dem  (errors=0)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'y', q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def z_lowercase():
    # expected: dem  (errors=0)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'z', q[1])
    d = mz(q[1])
    cudaq.detector(d)

# ============================================================================ #
# CATEGORY 2: Qubit scope variations
# ============================================================================ #

@cudaq.kernel
def same_qubit():
    # expected: dem  (errors=1)
    # Correct the qubit that was just measured.
    q = cudaq.qvector(1)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[0])
    cudaq.apply_noise(cudaq.XError, 0.1, q[0])
    d = mz(q[0])
    cudaq.detector(d)

@cudaq.kernel
def qvector_index():
    # expected: dem  (errors=1)
    # q[2] via constant subscript.
    q = cudaq.qvector(4)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[2])
    cudaq.apply_noise(cudaq.XError, 0.1, q[2])
    d = mz(q[2])
    cudaq.detector([m, d])

@cudaq.kernel
def standalone_qubit():
    # expected: dem  (errors=1)
    # Named cudaq.qubit rather than qvector element.
    a = cudaq.qubit()
    b = cudaq.qubit()
    h(a)
    m = mz(a)
    cudaq.conditioned_pauli_frame_update(m, 'X', b)
    cudaq.apply_noise(cudaq.XError, 0.1, b)
    d = mz(b)
    cudaq.detector([m, d])

# ============================================================================ #
# CATEGORY 3: Multiple corrections
# ============================================================================ #

@cudaq.kernel
def two_targets_one_meas():
    # expected: dem  (errors=2 detectors=2)
    # One measurement corrects two qubits.
    q = cudaq.qvector(3)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[2])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[2])
    d1 = mz(q[1])
    d2 = mz(q[2])
    cudaq.detector([m, d1])
    cudaq.detector([m, d2])

@cudaq.kernel
def two_meas_two_targets():
    # expected: dem  (errors=2 detectors=2)
    # Each measurement corrects its own qubit 
    q = cudaq.qvector(4)
    h(q[0])
    h(q[1])
    m0 = mz(q[0])
    m1 = mz(q[1])
    cudaq.conditioned_pauli_frame_update(m0, 'X', q[2])
    cudaq.conditioned_pauli_frame_update(m1, 'X', q[3])
    cudaq.apply_noise(cudaq.XError, 0.1, q[2])
    cudaq.apply_noise(cudaq.XError, 0.1, q[3])
    d2 = mz(q[2])
    d3 = mz(q[3])
    cudaq.detector([m0, d2])
    cudaq.detector([m1, d3])

@cudaq.kernel
def one_meas_five_targets():
    # expected: dem  (errors=1)
    # Syndrome corrects five data qubits simultaneously.
    q = cudaq.qvector(6)
    h(q[0])
    s = mz(q[0])
    for i in range(1, 6):
        cudaq.conditioned_pauli_frame_update(s, 'X', q[i])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    ds = mz(q[1])
    cudaq.detector([s, ds])

@cudaq.kernel
def mixed_paulis_one_meas():
    # expected: dem  (errors=1)
    # One measurement corrects X/Y/Z on different qubits.
    q = cudaq.qvector(4)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    cudaq.conditioned_pauli_frame_update(m, 'Y', q[2])
    cudaq.conditioned_pauli_frame_update(m, 'Z', q[3])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

# ============================================================================ #
# CATEGORY 4: Chained feedback (measure → correct → measure → correct ...)
# ============================================================================ #

@cudaq.kernel
def chain_2():
    # expected: dem  (errors=1)
    q = cudaq.qvector(1)
    h(q[0])
    m0 = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m0, 'X', q[0])
    m1 = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m1, 'X', q[0])
    cudaq.apply_noise(cudaq.XError, 0.1, q[0])
    d = mz(q[0])
    cudaq.detector(d)

@cudaq.kernel
def chain_5():
    # expected: dem  (errors=1)
    # Five successive rounds of measure-correct on the same qubit.
    q = cudaq.qvector(1)
    h(q[0])
    for _ in range(4):
        m = mz(q[0])
        cudaq.conditioned_pauli_frame_update(m, 'X', q[0])
    cudaq.apply_noise(cudaq.XError, 0.1, q[0])
    d = mz(q[0])
    cudaq.detector(d)

@cudaq.kernel
def chain_10():
    # expected: dem  (errors=1)
    # Ten-round correction chain.
    q = cudaq.qvector(1)
    h(q[0])
    for _ in range(9):
        m = mz(q[0])
        cudaq.conditioned_pauli_frame_update(m, 'X', q[0])
    cudaq.apply_noise(cudaq.XError, 0.1, q[0])
    d = mz(q[0])
    cudaq.detector(d)

# ============================================================================ #
# CATEGORY 5: Corrections interleaved with other gates
# ============================================================================ #

@cudaq.kernel
def correct_then_h_h():
    # expected: dem  (errors=1)
    # H; H = identity, so correction is still visible.
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    h(q[1])
    h(q[1])  # net identity
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def correct_then_cx():
    # expected: dem
    # Corrected qubit used as control in a CNOT.
    q = cudaq.qvector(3)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    x.ctrl(q[1], q[2])      # q[2] inherits the parity of q[1]
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d1 = mz(q[1])
    d2 = mz(q[2])
    cudaq.detector([m, d1])

@cudaq.kernel
def correct_then_another_meas():
    # expected: dem  (errors=1)
    # Corrected qubit remeasured and that second measure is the detector.
    q = cudaq.qvector(2)
    h(q[0])
    m0 = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m0, 'X', q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    m1 = mz(q[1])
    cudaq.conditioned_pauli_frame_update(m1, 'X', q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector(d)

@cudaq.kernel
def correct_then_reset_meas():
    # expected: dem  (errors=2 detectors=2)
    # After correction we reset the ancilla (not the data qubit), then reuse.
    q = cudaq.qvector(2)
    anc = cudaq.qubit()
    h(anc)
    s = mz(anc)
    cudaq.conditioned_pauli_frame_update(s, 'X', q[0])
    reset(anc)
    # Reuse ancilla for round 2
    h(anc)
    s2 = mz(anc)
    cudaq.conditioned_pauli_frame_update(s2, 'X', q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[0])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d0 = mz(q[0])
    d1 = mz(q[1])
    cudaq.detector([s, d0])
    cudaq.detector([s2, d1])

# ============================================================================ #
# CATEGORY 6: Teleportation and realistic QEC
# ============================================================================ #

@cudaq.kernel
def teleport():
    # expected: dem  (errors=1)
    q = cudaq.qvector(3)
    h(q[1])
    x.ctrl(q[1], q[2])  # Bell pair
    x.ctrl(q[0], q[1])
    h(q[0])  # Bell basis measurement of (q0, q1)
    m1 = mz(q[1])
    m0 = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m1, 'X', q[2])
    cudaq.conditioned_pauli_frame_update(m0, 'Z', q[2])
    cudaq.apply_noise(cudaq.XError, 0.05, q[2])
    d = mz(q[2])
    cudaq.detector(d)

@cudaq.kernel
def teleport_observable():
    # expected: dem  (errors=1)
    q = cudaq.qvector(3)
    h(q[1])
    x.ctrl(q[1], q[2])
    x.ctrl(q[0], q[1])
    h(q[0])
    m1 = mz(q[1])
    m0 = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m1, 'X', q[2])
    cudaq.conditioned_pauli_frame_update(m0, 'Z', q[2])
    cudaq.apply_noise(cudaq.XError, 0.05, q[2])
    r = mz(q[2])
    cudaq.logical_observable(r)

@cudaq.kernel
def rep_code_2round():
    # expected: dem  (errors>=1)
    # 2-round repetition code.  Ancillas are feedback-reset between rounds.
    data = cudaq.qvector(3)
    anc = cudaq.qvector(2)
    # Round 0
    x.ctrl(data[0], anc[0])
    x.ctrl(data[1], anc[0])
    x.ctrl(data[1], anc[1])
    x.ctrl(data[2], anc[1])
    s0a = mz(anc[0])
    s0b = mz(anc[1])
    cudaq.conditioned_pauli_frame_update(s0a, 'X', anc[0])  # reset ancilla to |0>
    cudaq.conditioned_pauli_frame_update(s0b, 'X', anc[1])
    cudaq.apply_noise(cudaq.XError, 0.05, data[1])
    # Round 1
    x.ctrl(data[0], anc[0])
    x.ctrl(data[1], anc[0])
    x.ctrl(data[1], anc[1])
    x.ctrl(data[2], anc[1])
    s1a = mz(anc[0])
    s1b = mz(anc[1])
    cudaq.detector([s0a, s1a])
    cudaq.detector([s0b, s1b])

@cudaq.kernel
def bit_flip_memory():
    # expected: dem  (errors>=1 detectors=2)
    # 2-round repetition-code memory: ZZ syndromes with feedback-reset ancillas.
    # data qubits start in |000>; correction resets ancillas between rounds.
    data = cudaq.qvector(3)
    anc0 = cudaq.qubit()
    anc1 = cudaq.qubit()
    # Round 0: ZZ syndromes
    x.ctrl(data[0], anc0)
    x.ctrl(data[1], anc0)
    x.ctrl(data[1], anc1)
    x.ctrl(data[2], anc1)
    s0a = mz(anc0)
    s0b = mz(anc1)
    cudaq.conditioned_pauli_frame_update(s0a, 'X', anc0)  # feedback-reset
    cudaq.conditioned_pauli_frame_update(s0b, 'X', anc1)
    # Inject error between rounds
    cudaq.apply_noise(cudaq.XError, 0.02, data[0])
    # Round 1: same syndromes
    x.ctrl(data[0], anc0)
    x.ctrl(data[1], anc0)
    x.ctrl(data[1], anc1)
    x.ctrl(data[2], anc1)
    s1a = mz(anc0)
    s1b = mz(anc1)
    cudaq.detector([s0a, s1a])
    cudaq.detector([s0b, s1b])

# ============================================================================ #
# CATEGORY 7: Edge cases - no noise (errors=0 DEM)
# ============================================================================ #

@cudaq.kernel
def no_noise_x():
    # expected: dem  (errors=0)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def no_noise_chain():
    # expected: dem  (errors=0)
    q = cudaq.qvector(1)
    h(q[0])
    for _ in range(5):
        m = mz(q[0])
        cudaq.conditioned_pauli_frame_update(m, 'X', q[0])
    d = mz(q[0])
    cudaq.detector(d)

# ============================================================================ #
# CATEGORY 8: Large circuits
# ============================================================================ #

@cudaq.kernel
def large_20qubits():
    # expected: dem  (errors=1)
    # Syndrome qubit corrects 19 data qubits.
    q = cudaq.qvector(20)
    h(q[0])
    s = mz(q[0])
    for i in range(1, 20):
        cudaq.conditioned_pauli_frame_update(s, 'X', q[i])
    cudaq.apply_noise(cudaq.XError, 0.01, q[1])
    d = mz(q[1])
    cudaq.detector([s, d])

@cudaq.kernel
def many_corrections_many_detectors():
    # expected: dem  (errors=6 detectors=6)
    # 6 syndrome qubits each correcting one data qubit (6 detectors).
    # Uses explicit variable names to avoid Python list inside kernel.
    q = cudaq.qvector(12)
    h(q[0])
    m0 = mz(q[0])
    h(q[1])
    m1 = mz(q[1])
    h(q[2])
    m2 = mz(q[2])
    h(q[3])
    m3 = mz(q[3])
    h(q[4])
    m4 = mz(q[4])
    h(q[5])
    m5 = mz(q[5])
    cudaq.conditioned_pauli_frame_update(m0, 'X', q[6])
    cudaq.conditioned_pauli_frame_update(m1, 'X', q[7])
    cudaq.conditioned_pauli_frame_update(m2, 'X', q[8])
    cudaq.conditioned_pauli_frame_update(m3, 'X', q[9])
    cudaq.conditioned_pauli_frame_update(m4, 'X', q[10])
    cudaq.conditioned_pauli_frame_update(m5, 'X', q[11])
    cudaq.apply_noise(cudaq.XError, 0.05, q[6])
    cudaq.apply_noise(cudaq.XError, 0.05, q[7])
    cudaq.apply_noise(cudaq.XError, 0.05, q[8])
    cudaq.apply_noise(cudaq.XError, 0.05, q[9])
    cudaq.apply_noise(cudaq.XError, 0.05, q[10])
    cudaq.apply_noise(cudaq.XError, 0.05, q[11])
    d0 = mz(q[6])
    d1 = mz(q[7])
    d2 = mz(q[8])
    d3 = mz(q[9])
    d4 = mz(q[10])
    d5 = mz(q[11])
    cudaq.detector([m0, d0])
    cudaq.detector([m1, d1])
    cudaq.detector([m2, d2])
    cudaq.detector([m3, d3])
    cudaq.detector([m4, d4])
    cudaq.detector([m5, d5])

# ============================================================================ #
# CATEGORY 9: Interaction with cudaq.detectors (pair form)
# ============================================================================ #

@cudaq.kernel
def with_detectors_pair():
    # expected: dem
    # Cross-round ZZ parity detectors via cudaq.detectors(m_prev, m_curr).
    # Two ancillas measure parity of data[i] XOR data[i+1].
    # Feedback-reset ancillas between rounds using conditioned_pauli_frame_update
    # on individual handles indexed from the mz-vector (m_prev[i]).
    data = cudaq.qvector(3)
    anc  = cudaq.qvector(2)
    # Round 0
    x.ctrl(data[0], anc[0])
    x.ctrl(data[1], anc[0])
    x.ctrl(data[1], anc[1])
    x.ctrl(data[2], anc[1])
    m_prev = mz(anc)
    cudaq.conditioned_pauli_frame_update(m_prev[0], 'X', anc[0])  # ancilla reset
    cudaq.conditioned_pauli_frame_update(m_prev[1], 'X', anc[1])
    # Inject error on data[0] between rounds
    cudaq.apply_noise(cudaq.XError, 0.02, data[0])
    # Round 1
    x.ctrl(data[0], anc[0])
    x.ctrl(data[1], anc[0])
    x.ctrl(data[1], anc[1])
    x.ctrl(data[2], anc[1])
    m_curr = mz(anc)
    cudaq.detectors(m_prev, m_curr)

# ============================================================================ #
# CATEGORY 10: Noise model variations
# ============================================================================ #

@cudaq.kernel
def y_error_with_x_feedback():
    # expected: dem  (errors=1)
    # X feedback corrects bit-flip component of Y error.
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    cudaq.apply_noise(cudaq.YError, 0.1, q[1])  # Y = iXZ
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def z_error_invisible():
    # expected: dem  (errors=0)
    # Z error is invisible in Z-basis measurement: X feedback + Z error -> errors=0.
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    cudaq.apply_noise(cudaq.ZError, 0.1, q[1])  # phase flip, invisible to mz
    d = mz(q[1])
    cudaq.detector([m, d])

# ============================================================================ #
# CATEGORY 11: Two explicit corrections (different measurements) on one qubit
# ============================================================================ #

@cudaq.kernel
def mix_explicit_and_implicit():
    # expected: dem  (errors=1)
    # Two explicit frame updates on the same target qubit, conditioned on two
    # distinct measurements: X on m0, Z on m1. The Z feedback is invisible to
    # the Z-basis detector, so only the X_ERROR shows up (errors=1).
    q = cudaq.qvector(3)
    h(q[0])
    h(q[1])
    m0 = mz(q[0])
    m1 = mz(q[1])
    cudaq.conditioned_pauli_frame_update(m0, 'X', q[2])
    cudaq.conditioned_pauli_frame_update(m1, 'Z', q[2])
    cudaq.apply_noise(cudaq.XError, 0.1, q[2])
    d = mz(q[2])
    cudaq.detector([m0, d])

# ============================================================================ #
# CATEGORY 12: Non-deterministic (should be rejected by Stim)
# ============================================================================ #

@cudaq.kernel
def double_x_cancel():
    # expected: nondet
    # CX rec[-1] q then CX rec[-1] q cancels -> detector is m XOR 0 = m (nondet).
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])  # cancels the first
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])  # detector is now nondet: m XOR 0 = m

@cudaq.kernel
def z_fb_bad_detector():
    # expected: nondet
    # Z doesn't flip Z-basis, so including m in detector makes it nondet.
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'Z', q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])  # m XOR (X-error outcome) is nondet

@cudaq.kernel
def reset_wipes_correction():
    # expected: nondet
    # Correction applied, then the qubit is reset -> correction lost.
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    reset(q[1])                               # wipes the feedback state
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])  # nondet: d sees only noise, not m

@cudaq.kernel
def correction_wrong_qubit():
    # expected: nondet
    # Correct q[1] but detector is on q[2]: q[2] has no correction.
    q = cudaq.qvector(3)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])  # corrects q[1]
    cudaq.apply_noise(cudaq.XError, 0.1, q[2])
    d = mz(q[2])
    cudaq.detector([m, d])  # q[2] is nondet

# ============================================================================ #
# CATEGORY 13: Compile-time errors (bad pauli argument)
# ============================================================================ #

@cudaq.kernel
def bad_pauli_Q():
    # expected: err_compile
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'Q', q[1])  # invalid
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def bad_pauli_A():
    # expected: err_compile
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'A', q[1])  # invalid
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def bad_pauli_I():
    # expected: err_compile
    # 'I' is a valid Pauli but we don't support identity correction.
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'I', q[1])  # not a supported Pauli
    d = mz(q[1])
    cudaq.detector([m, d])

# ============================================================================ #
# CATEGORY 14: Tricky interaction - Z error invisible to X-basis detector
# ============================================================================ #

@cudaq.kernel
def x_error_detectable():
    # expected: dem  (errors=1)
    # Confirms that X correction makes X errors detectable (D0 present).
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    cudaq.apply_noise(cudaq.XError, 0.15, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def z_error_not_detectable():
    # expected: dem  (errors=0)
    # Z error is invisible to mz: DEM should have errors=0 even with ZError.
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    cudaq.conditioned_pauli_frame_update(m, 'X', q[1])
    cudaq.apply_noise(cudaq.ZError, 0.15, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])


# ============================================================================ #
# Master suite: (name, kernel, args, expected, optional_note)
# ============================================================================ #
SUITE = [
    # --- Category 1: basic Paulis and case ---
    ("x_basic",              x_basic,              (), "dem",          "errors=1"),
    ("y_basic",              y_basic,              (), "dem",          "errors=1"),
    ("z_basic",              z_basic,              (), "dem",          "errors=1"),
    ("x_lowercase",          x_lowercase,          (), "dem",          "errors=0"),
    ("y_lowercase",          y_lowercase,          (), "dem",          "errors=0"),
    ("z_lowercase",          z_lowercase,          (), "dem",          "errors=0"),
    # --- Category 2: qubit scope ---
    ("same_qubit",           same_qubit,           (), "dem",          "errors=1"),
    ("qvector_index",        qvector_index,        (), "dem",          "errors=1"),
    ("standalone_qubit",     standalone_qubit,     (), "dem",          "errors=1"),
    # --- Category 3: multiple corrections ---
    ("two_targets_one_meas", two_targets_one_meas, (), "dem",          "errors=2 detectors=2"),
    ("two_meas_two_targets", two_meas_two_targets, (), "dem",          "errors=2 detectors=2"),
    ("one_meas_five_targets",one_meas_five_targets,(), "dem",          "errors=1"),
    ("mixed_paulis_one_meas",mixed_paulis_one_meas,(), "dem",          "errors=1"),
    # --- Category 4: chained ---
    ("chain_2",              chain_2,              (), "dem",          "errors=1"),
    ("chain_5",              chain_5,              (), "dem",          "errors=1"),
    ("chain_10",             chain_10,             (), "dem",          "errors=1"),
    # --- Category 5: interleaved ops ---
    ("correct_then_h_h",     correct_then_h_h,     (), "dem",          "errors=1"),
    ("correct_then_cx",      correct_then_cx,      (), "dem",          None),
    ("correct_then_another_meas", correct_then_another_meas, (), "dem", "errors=1"),
    ("correct_then_reset_meas",   correct_then_reset_meas,   (), "dem", "errors=2 detectors=2"),
    # --- Category 6: realistic QEC ---
    ("teleport",             teleport,             (), "dem",          "errors=1"),
    ("teleport_observable",  teleport_observable,  (), "dem",          "errors=1"),
    ("rep_code_2round",      rep_code_2round,      (), "dem",          "errors>=1"),
    ("bit_flip_memory",      bit_flip_memory,      (), "dem",          "errors>=1 detectors=2"),
    # --- Category 7: no noise ---
    ("no_noise_x",           no_noise_x,           (), "dem",          "errors=0"),
    ("no_noise_chain",       no_noise_chain,       (), "dem",          "errors=0"),
    # --- Category 8: large ---
    ("large_20qubits",       large_20qubits,       (), "dem",          "errors=1"),
    ("many_corrections_many_detectors", many_corrections_many_detectors, (), "dem", "errors=6 detectors=6"),
    # --- Category 9: detectors pair ---
    ("with_detectors_pair",  with_detectors_pair,  (), "dem",          None),
    # --- Category 10: noise variations ---
    ("y_error_with_x_feedback",  y_error_with_x_feedback,  (), "dem",  "errors=1"),
    ("z_error_invisible",        z_error_invisible,        (), "dem",  "errors=0"),
    # --- Category 11: mix explicit + implicit ---
    ("mix_explicit_and_implicit",mix_explicit_and_implicit, (), "dem", "errors=1"),
    # --- Category 12: non-deterministic ---
    ("double_x_cancel",      double_x_cancel,      (), "nondet",       None),
    ("z_fb_bad_detector",    z_fb_bad_detector,    (), "nondet",       None),
    ("reset_wipes_correction",reset_wipes_correction, (), "nondet",    None),
    ("correction_wrong_qubit",correction_wrong_qubit, (), "nondet",    None),
    # --- Category 13: compile-time bad pauli ---
    ("bad_pauli_Q",          bad_pauli_Q,          (), "err_compile",  None),
    ("bad_pauli_A",          bad_pauli_A,          (), "err_compile",  None),
    ("bad_pauli_I",          bad_pauli_I,          (), "err_compile",  None),
    # --- Category 14: error visibility ---
    ("x_error_detectable",   x_error_detectable,  (), "dem",          "errors=1"),
    ("z_error_not_detectable",z_error_not_detectable,(),"dem",        "errors=0"),
]

_B1_MSG  = "cannot be represented as symbolic Pauli"
_NDT_MSG = "non-deterministic"


def count(dem_text, substring):
    return dem_text.count(substring)


def classify(kernel, args):
    try:
        dem = cudaq.dem_from_kernel(kernel, *args, noise_model=nm).strip()
        nerrors = count(dem, "error(")
        ndets   = count(dem, " D")
        nobs    = count(dem, " L")
        return "dem", f"errors={nerrors} detectors={ndets} observables={nobs}"
    except Exception as e:
        msg = str(e).splitlines()[0]
        if _B1_MSG in str(e):
            return "reject_b1", msg[:60]
        if _NDT_MSG in msg or "non-deterministic" in msg:
            return "nondet", msg[:60]
        # Compile-time fatal from the bridge (bad pauli char, wrong type)
        if ("pauli must be" in msg or "requires exactly 3" in msg or
                "compile-time constant" in msg or "Fatal" in msg or
                "RuntimeError" in msg):
            return "err_compile", msg[:60]
        return "err_compile", msg[:60]   # anything else during kernel compile


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None

    npass = nfail = 0
    for entry in SUITE:
        name, kernel, args, expected, note = entry
        if only and name != only:
            continue
        outcome, detail = classify(kernel, args)
        ok = (outcome == expected)
        npass += ok
        nfail += (not ok)
        flag = "PASS" if ok else f"FAIL(exp {expected})"
        note_str = f"  [{note}]" if note else ""
        print(f"  [{flag:22s}] {name:30s} {outcome:14s} {detail}{note_str}")

    print(f"\n{npass} passed, {nfail} failed")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
