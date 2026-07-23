#!/usr/bin/env python3
# ============================================================================ #
# Expected outcomes for tests:
#   "dem"        -> cudaq.dem_from_kernel yields a valid DEM
#   "nondet"     -> Stim rejects a non-deterministic detector
#   "err_compile"  -> the condition/op is not representable as symbolic feedback
#                   (non-Pauli/controlled gate, non-linear `and`/`or`, ...)
# ============================================================================ #

import cudaq, sys

nm = cudaq.NoiseModel()   # empty noise model - still tracks noise ops inside kernels

# ============================================================================ #
# CATEGORY 1: All three Paulis (case-insensitivity is moot in gate form)
# ============================================================================ #

@cudaq.kernel
def x_basic():
    # expected: dem  (errors=1)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    if m:
        x(q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def y_basic():
    # expected: dem  (errors=1)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    if m:
        y(q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def z_basic():
    # expected: dem  (errors=1)
    # Z|0> = |0>: no bit flip, so detector on d alone (not including m) is valid.
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    if m:
        z(q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector(d)

@cudaq.kernel
def x_no_noise():
    # expected: dem  (errors=0)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    if m:
        x(q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def y_no_noise():
    # expected: dem  (errors=0)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    if m:
        y(q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def z_no_noise():
    # expected: dem  (errors=0)
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    if m:
        z(q[1])
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
    if m:
        x(q[0])
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
    if m:
        x(q[2])
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
    if m:
        x(b)
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
    if m:
        x(q[1])
        x(q[2])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[2])
    d1 = mz(q[1])
    d2 = mz(q[2])
    cudaq.detector([m, d1])
    cudaq.detector([m, d2])

@cudaq.kernel
def two_meas_two_targets():
    # expected: dem  (errors=2 detectors=2)
    # Each measurement corrects its own qubit (standard repetition pattern).
    q = cudaq.qvector(4)
    h(q[0])
    h(q[1])
    m0 = mz(q[0])
    m1 = mz(q[1])
    if m0:
        x(q[2])
    if m1:
        x(q[3])
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
        if s:
            x(q[i])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([s, d])

@cudaq.kernel
def mixed_paulis_one_meas():
    # expected: dem  (errors=1)
    # One measurement corrects X/Y/Z on different qubits.
    q = cudaq.qvector(4)
    h(q[0])
    m = mz(q[0])
    if m:
        x(q[1])
        y(q[2])
        z(q[3])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

# ============================================================================ #
# CATEGORY 4: Chained feedback (measure -> correct -> measure -> correct ...)
# ============================================================================ #

@cudaq.kernel
def chain_2():
    # expected: dem  (errors=1)
    q = cudaq.qvector(1)
    h(q[0])
    m0 = mz(q[0])
    if m0:
        x(q[0])
    m1 = mz(q[0])
    if m1:
        x(q[0])
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
        if m:
            x(q[0])
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
        if m:
            x(q[0])
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
    if m:
        x(q[1])
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
    if m:
        x(q[1])
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
    if m0:
        x(q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    m1 = mz(q[1])
    if m1:
        x(q[1])
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
    if s:
        x(q[0])
    reset(anc)
    # Reuse ancilla for round 2
    h(anc)
    s2 = mz(anc)
    if s2:
        x(q[1])
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
    if m1:
        x(q[2])
    if m0:
        z(q[2])
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
    if m1:
        x(q[2])
    if m0:
        z(q[2])
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
    if s0a:
        x(anc[0])  # reset ancilla to |0>
    if s0b:
        x(anc[1])
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
    if s0a:
        x(anc0)  # feedback-reset
    if s0b:
        x(anc1)
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
    if m:
        x(q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def no_noise_chain():
    # expected: dem  (errors=0)
    q = cudaq.qvector(1)
    h(q[0])
    for _ in range(5):
        m = mz(q[0])
        if m:
            x(q[0])
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
        if s:
            x(q[i])
    cudaq.apply_noise(cudaq.XError, 0.01, q[1])
    d = mz(q[1])
    cudaq.detector([s, d])

@cudaq.kernel
def many_corrections_many_detectors():
    # expected: dem  (errors=6 detectors=6)
    # 6 syndrome qubits each correcting one data qubit (6 detectors).
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
    if m0:
        x(q[6])
    if m1:
        x(q[7])
    if m2:
        x(q[8])
    if m3:
        x(q[9])
    if m4:
        x(q[10])
    if m5:
        x(q[11])
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
    # Feedback-reset ancillas between rounds, conditioning on individual handles
    # indexed from the mz-vector (m_prev[i]).
    data = cudaq.qvector(3)
    anc = cudaq.qvector(2)
    # Round 0
    x.ctrl(data[0], anc[0])
    x.ctrl(data[1], anc[0])
    x.ctrl(data[1], anc[1])
    x.ctrl(data[2], anc[1])
    m_prev = mz(anc)
    if m_prev[0]:
        x(anc[0])  # ancilla reset
    if m_prev[1]:
        x(anc[1])
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
    if m:
        x(q[1])
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
    if m:
        x(q[1])
    cudaq.apply_noise(cudaq.ZError, 0.1, q[1])  # phase flip, invisible to mz
    d = mz(q[1])
    cudaq.detector([m, d])

# ============================================================================ #
# CATEGORY 11: Two corrections (different measurements) on one qubit
# ============================================================================ #

@cudaq.kernel
def mix_x_and_z():
    # expected: dem  (errors=1)
    # X on m0, Z on m1 on the same target. The Z feedback is invisible to the
    # Z-basis detector, so only the X_ERROR shows up (errors=1).
    q = cudaq.qvector(3)
    h(q[0])
    h(q[1])
    m0 = mz(q[0])
    m1 = mz(q[1])
    if m0:
        x(q[2])
    if m1:
        z(q[2])
    cudaq.apply_noise(cudaq.XError, 0.1, q[2])
    d = mz(q[2])
    cudaq.detector([m0, d])

# ============================================================================ #
# CATEGORY 12: Non-deterministic (should be rejected by Stim)
# ============================================================================ #

@cudaq.kernel
def double_x_cancel():
    # expected: nondet
    # Two X corrections on m cancel -> detector is m XOR 0 = m (nondet).
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    if m:
        x(q[1])
        x(q[1])  # cancels the first
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
    if m:
        z(q[1])
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
    if m:
        x(q[1])
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
    if m:
        x(q[1])  # corrects q[1]
    cudaq.apply_noise(cudaq.XError, 0.1, q[2])
    d = mz(q[2])
    cudaq.detector([m, d])  # q[2] is nondet

# ============================================================================ #
# CATEGORY 13: Not representable as feedback (rejected up front by the pass/guard)
#
# The `bad_pauli_*` primitive-argument tests have no analog in the `if (m) P(q)`
# form -- there is no Pauli character to validate. Their equivalent here is the
# class of conditional operations the pass CANNOT lower to symbolic feedback.
# ============================================================================ #

@cudaq.kernel
def cond_nonpauli():
    # expected: err_compile  (non-Pauli gate)
    # if (m) H(q): H is not a Pauli -> no feedback representation.
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    if m:
        h(q[1])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])

@cudaq.kernel
def cond_controlled():
    # expected: err_compile  (controlled gate)
    # if (m) CX(q1, q2): controlled gate -> not a single-qubit Pauli.
    q = cudaq.qvector(3)
    h(q[0])
    m = mz(q[0])
    if m:
        x.ctrl(q[1], q[2])
    cudaq.apply_noise(cudaq.XError, 0.1, q[2])
    d = mz(q[2])
    cudaq.detector([m, d])

@cudaq.kernel
def cond_compound_and():
    # expected: err_compile  (nonlinear AND)
    # if (m0 and m1) X(q): non-linear (AND) condition -> not affine feedback.
    q = cudaq.qvector(3)
    h(q[0])
    h(q[1])
    m0 = mz(q[0])
    m1 = mz(q[1])
    if m0 and m1:
        x(q[2])
    cudaq.apply_noise(cudaq.XError, 0.1, q[2])
    d = mz(q[2])
    cudaq.detector([m0, m1, d])

# ============================================================================ #
# CATEGORY 14: Error visibility
# ============================================================================ #

@cudaq.kernel
def x_error_detectable():
    # expected: dem  (errors=1)
    # X correction makes X errors detectable (D0 present).
    q = cudaq.qvector(2)
    h(q[0])
    m = mz(q[0])
    if m:
        x(q[1])
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
    if m:
        x(q[1])
    cudaq.apply_noise(cudaq.ZError, 0.15, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])


# ############################################################################ #
# ############################################################################ #
#                                                                              #
#  RICHER IF-BLOCKS -- extra work in/around the branch that STILL lowers to a  #
#  valid inlined-feedback DEM.  The pass needs the branch body to reduce to    #
#  single-qubit Pauli gates, so anything that OPTIMIZES AWAY to Paulis works:  #
#  loops unroll, constant `if`s fold, pure index arithmetic is hoisted, and    #
#  extra gates/resets/allocations OUTSIDE the branch are fine.  (A bare non-   #
#  Pauli gate or `reset` left INSIDE the branch is instead rejected -- see     #
#  cond_nonpauli / cond_controlled above.)                                     #
#                                                                              #
# ############################################################################ #
# ############################################################################ #

@cudaq.kernel
def rich_loop_in_branch():
    # expected: dem  (errors=1)
    # A for-loop of corrections inside the branch, unrolled to Paulis.
    q = cudaq.qvector(5)
    h(q[0])
    s = mz(q[0])
    if s:
        for i in range(1, 5):
            x(q[i])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([s, d])

@cudaq.kernel
def rich_nested_loop_mixed():
    # expected: dem  (errors=1)
    # Nested loops with mixed X/Z corrections and computed indices in the branch.
    q = cudaq.qvector(7)
    h(q[0])
    s = mz(q[0])
    if s:
        for i in range(3):
            x(q[1 + i])
        for j in range(3):
            z(q[4 + j])
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([s, d])

@cudaq.kernel
def rich_pure_logic_in_branch():
    # expected: dem  (errors=1)
    # A constant-folded nested `if` and pure index arithmetic inside the branch.
    q = cudaq.qvector(4)
    h(q[0])
    m = mz(q[0])
    if m:
        k = 1 + 1
        if k > 1:
            x(q[k])
    cudaq.apply_noise(cudaq.XError, 0.1, q[2])
    d = mz(q[2])
    cudaq.detector([m, d])

@cudaq.kernel
def rich_surrounding_work():
    # expected: dem  (errors=1)
    # A 2nd qvector, an ancilla reset and an entangling gate -- extra work
    # OUTSIDE the branch (on other qubits) -- around a clean feedback correction.
    data = cudaq.qvector(2)
    anc = cudaq.qvector(1)
    h(data[0])
    m = mz(data[0])
    if m:
        x(data[1])
    reset(anc[0])
    h(anc[0])
    x.ctrl(anc[0], data[0])
    cudaq.apply_noise(cudaq.XError, 0.1, data[1])
    d = mz(data[1])
    cudaq.detector([m, d])

@cudaq.kernel
def rich_round_loop():
    # expected: dem  (errors=1)
    # A loop of feedback-correction rounds (nested logic); a final error after
    # the last correction is caught by the detector.
    data = cudaq.qvector(1)
    h(data[0])
    m = mz(data[0])
    if m:
        x(data[0])
    for r in range(3):
        cudaq.apply_noise(cudaq.XError, 0.05, data[0])
        cur = mz(data[0])
        if cur:
            x(data[0])
    cudaq.apply_noise(cudaq.XError, 0.1, data[0])
    d = mz(data[0])
    cudaq.detector(d)


# ############################################################################ #
# ############################################################################ #
#                                                                              #
#  NESTED KERNELS -- feedback `if (m) P(q)` living inside a CALLED sub-kernel. #
#  The pass is per-function and the DEM reject-guard is whole-module, so       #
#  feedback lowers wherever it lives: inside a callee, through two levels of   #
#  nesting, and across repeated invocations.  The only limit is detector       #
#  SCOPE -- a measurement made inside a callee that never escapes can't be     #
#  cancelled by a parent detector (nested_diff_qubit is a correct nondet).     #
#                                                                              #
# ############################################################################ #
# ############################################################################ #

@cudaq.kernel
def _nested_self_correct(q: cudaq.qview):
    m = mz(q[0])
    if m:
        x(q[0])

@cudaq.kernel
def nested_self_contained():
    # expected: dem  (errors=1)
    # Feedback fully self-contained in the callee (measure + correct same qubit).
    q = cudaq.qvector(1)
    h(q[0])
    _nested_self_correct(q)
    cudaq.apply_noise(cudaq.XError, 0.1, q[0])
    d = mz(q[0])
    cudaq.detector(d)

@cudaq.kernel
def _nested_syndrome_correct(q: cudaq.qview):
    s = mz(q[0])
    if s:
        x(q[1])

@cudaq.kernel
def nested_diff_qubit():
    # expected: nondet
    # Callee measures an ancilla and corrects a different data qubit; that
    # ancilla measurement never escapes, so the parent detector can't cancel it.
    q = cudaq.qvector(2)
    h(q[0])
    _nested_syndrome_correct(q)
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector(d)

@cudaq.kernel
def _nested_round(q: cudaq.qview):
    m = mz(q[0])
    if m:
        x(q[0])

@cudaq.kernel
def nested_rounds():
    # expected: dem  (errors=1)
    # The feedback callee is invoked repeatedly (rounds via calls).
    q = cudaq.qvector(1)
    h(q[0])
    for _ in range(3):
        _nested_round(q)
        cudaq.apply_noise(cudaq.XError, 0.05, q[0])
    d = mz(q[0])
    cudaq.detector(d)

@cudaq.kernel
def _nested_leaf(q: cudaq.qview):
    m = mz(q[0])
    if m:
        x(q[0])

@cudaq.kernel
def _nested_mid(q: cudaq.qview):
    _nested_leaf(q)

@cudaq.kernel
def nested_two_level():
    # expected: dem  (errors=1)
    # Feedback in the leaf lowers through two levels (parent -> mid -> leaf).
    q = cudaq.qvector(1)
    h(q[0])
    _nested_mid(q)
    cudaq.apply_noise(cudaq.XError, 0.1, q[0])
    d = mz(q[0])
    cudaq.detector(d)

@cudaq.kernel
def _nested_correct_other(q: cudaq.qview):
    m = mz(q[0])
    if m:
        x(q[1])

@cudaq.kernel
def nested_detector_pair():
    # expected: dem  (errors=1)
    # Callee feedback ties to the PARENT's own measurement: detector([m_top, d])
    # collapses to `err` (deterministic).
    q = cudaq.qvector(2)
    h(q[0])
    m_top = mz(q[0])
    _nested_correct_other(q)
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m_top, d])

@cudaq.kernel
def _nested_measure_ret(q: cudaq.qview) -> cudaq.measure_handle:
    m = mz(q[0])
    if m:
        x(q[1])
    return m           # returning a measure_handle (not bool) keeps it usable

@cudaq.kernel
def nested_return_meas():
    # expected: dem  (errors=1)
    # Callee measures + corrects a different qubit and RETURNS its measure_handle
    # so the parent can reference it in a detector; detector([m, d]) collapses to
    # `err`. Requires the `-> cudaq.measure_handle` return type -- returning a
    # plain `bool` is rejected as a non-homogenous detector list.
    q = cudaq.qvector(2)
    h(q[0])
    m = _nested_measure_ret(q)
    cudaq.apply_noise(cudaq.XError, 0.1, q[1])
    d = mz(q[1])
    cudaq.detector([m, d])


# ============================================================================ #
# Master suite: (name, kernel, args, expected, optional_note)
# ============================================================================ #
SUITE = [
    # --- Category 1: basic Paulis ---
    ("x_basic",              x_basic,              (), "dem",       "errors=1"),
    ("y_basic",              y_basic,              (), "dem",       "errors=1"),
    ("z_basic",              z_basic,              (), "dem",       "errors=1"),
    ("x_no_noise",           x_no_noise,           (), "dem",       "errors=0"),
    ("y_no_noise",           y_no_noise,           (), "dem",       "errors=0"),
    ("z_no_noise",           z_no_noise,           (), "dem",       "errors=0"),
    # --- Category 2: qubit scope ---
    ("same_qubit",           same_qubit,           (), "dem",       "errors=1"),
    ("qvector_index",        qvector_index,        (), "dem",       "errors=1"),
    ("standalone_qubit",     standalone_qubit,     (), "dem",       "errors=1"),
    # --- Category 3: multiple corrections ---
    ("two_targets_one_meas", two_targets_one_meas, (), "dem",       "errors=2 detectors=2"),
    ("two_meas_two_targets", two_meas_two_targets, (), "dem",       "errors=2 detectors=2"),
    ("one_meas_five_targets",one_meas_five_targets,(), "dem",       "errors=1"),
    ("mixed_paulis_one_meas",mixed_paulis_one_meas,(), "dem",       "errors=1"),
    # --- Category 4: chained ---
    ("chain_2",              chain_2,              (), "dem",       "errors=1"),
    ("chain_5",              chain_5,              (), "dem",       "errors=1"),
    ("chain_10",             chain_10,             (), "dem",       "errors=1"),
    # --- Category 5: interleaved ops ---
    ("correct_then_h_h",     correct_then_h_h,     (), "dem",       "errors=1"),
    ("correct_then_cx",      correct_then_cx,      (), "dem",       None),
    ("correct_then_another_meas", correct_then_another_meas, (), "dem", "errors=1"),
    ("correct_then_reset_meas",   correct_then_reset_meas,   (), "dem", "errors=2 detectors=2"),
    # --- Category 6: realistic QEC ---
    ("teleport",             teleport,             (), "dem",       "errors=1"),
    ("teleport_observable",  teleport_observable,  (), "dem",       "errors=1"),
    ("rep_code_2round",      rep_code_2round,      (), "dem",       "errors>=1"),
    ("bit_flip_memory",      bit_flip_memory,      (), "dem",       "errors>=1 detectors=2"),
    # --- Category 7: no noise ---
    ("no_noise_x",           no_noise_x,           (), "dem",       "errors=0"),
    ("no_noise_chain",       no_noise_chain,       (), "dem",       "errors=0"),
    # --- Category 8: large ---
    ("large_20qubits",       large_20qubits,       (), "dem",       "errors=1"),
    ("many_corrections_many_detectors", many_corrections_many_detectors, (), "dem", "errors=6 detectors=6"),
    # --- Category 9: detectors pair ---
    ("with_detectors_pair",  with_detectors_pair,  (), "dem",       None),
    # --- Category 10: noise variations ---
    ("y_error_with_x_feedback",  y_error_with_x_feedback,  (), "dem", "errors=1"),
    ("z_error_invisible",        z_error_invisible,        (), "dem", "errors=0"),
    # --- Category 11: two corrections on one qubit ---
    ("mix_x_and_z",          mix_x_and_z,          (), "dem",       "errors=1"),
    # --- Category 12: non-deterministic ---
    ("double_x_cancel",      double_x_cancel,      (), "nondet",    None),
    ("z_fb_bad_detector",    z_fb_bad_detector,    (), "nondet",    None),
    ("reset_wipes_correction",reset_wipes_correction, (), "nondet", None),
    ("correction_wrong_qubit",correction_wrong_qubit, (), "nondet", None),
    # --- Category 13: not representable as feedback -> pass/guard rejects ---
    ("cond_nonpauli",        cond_nonpauli,        (), "err_compile", "non-Pauli gate"),
    ("cond_controlled",      cond_controlled,      (), "err_compile", "controlled gate"),
    ("cond_compound_and",    cond_compound_and,    (), "err_compile", "nonlinear AND"),
    # --- Category 14: error visibility ---
    ("x_error_detectable",   x_error_detectable,   (), "dem",       "errors=1"),
    ("z_error_not_detectable",z_error_not_detectable,(),"dem",      "errors=0"),
    # --- Richer if-blocks that still lower to valid inlined feedback ---
    ("rich_loop_in_branch",       rich_loop_in_branch,       (), "dem", "errors=1"),
    ("rich_nested_loop_mixed",    rich_nested_loop_mixed,    (), "dem", "errors=1"),
    ("rich_pure_logic_in_branch", rich_pure_logic_in_branch, (), "dem", "errors=1"),
    ("rich_surrounding_work",     rich_surrounding_work,     (), "dem", "errors=1"),
    ("rich_round_loop",           rich_round_loop,           (), "dem", "errors=1"),
    # --- Nested kernels: feedback inside a called sub-kernel ---
    ("nested_self_contained",     nested_self_contained,     (), "dem",    "errors=1"),
    ("nested_diff_qubit",         nested_diff_qubit,         (), "nondet", None),
    ("nested_rounds",             nested_rounds,             (), "dem",    "errors=1"),
    ("nested_two_level",          nested_two_level,          (), "dem",    "errors=1"),
    ("nested_detector_pair",      nested_detector_pair,      (), "dem",    "errors=1"),
    ("nested_return_meas",        nested_return_meas,        (), "dem",    "errors=1"),
]

_B1_MSG  = "cannot be represented as symbolic Pauli"
_NDT_MSG = "non-deterministic"


def classify(kernel, args):
    try:
        dem = cudaq.dem_from_kernel(kernel, *args, noise_model=nm).strip()
        nerrors = dem.count("error(")
        ndets = dem.count(" D")
        nobs = dem.count(" L")
        return "dem", f"errors={nerrors} detectors={ndets} observables={nobs}"
    except Exception as e:
        msg = str(e).splitlines()[0]
        if _B1_MSG in str(e):
            return "err_compile", msg[:55]
        if _NDT_MSG in msg:
            return "nondet", msg[:55]
        return "other", msg[:55]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    npass = nfail = 0
    for name, kernel, args, expected, note in SUITE:
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
