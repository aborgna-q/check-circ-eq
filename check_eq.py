from pytket import Circuit
from pytket.extensions.cutensornet import TensorNetwork
import cuquantum as cq
import numpy as np

def test_equivalence(circ1, circ2):
    n_qubits = circ1.n_qubits
    assert n_qubits == circ2.n_qubits

    bell_states = Circuit(2*n_qubits)
    for q in range(n_qubits):
        bell_states.H(q)
    for q in range(n_qubits):
        bell_states.CX(q, q+n_qubits)

    ket_circ = bell_states.copy()
    ket_circ.add_circuit(circ1, qubits=[q for q in range(n_qubits)])
    bra_circ = bell_states
    bra_circ.add_circuit(circ2, qubits=[q for q in range(n_qubits)])

    # Create the TNs of the circuits
    ket_net = TensorNetwork(ket_circ)
    bra_net = TensorNetwork(bra_circ)
    # Concatenate one with the other, netB is the adjoint
    overlap_net = ket_net.vdot(bra_net)
    # Run the contraction
    overlap = cq.contract(*overlap_net)

    return np.isclose(overlap, 1)

if __name__ == "__main__":
    import os
    import json
    import time

    old_circs = 'bef'
    new_circs = 'aft'

    n_skipped = 0
    n_success = 0
    n_fail = 0

    MAX_QUBITS = 40

    print("Starting")

    for filename in os.listdir(new_circs):
        if filename.startswith('final_circ_'):
            name = filename[11:]
            new_circ_f = os.path.join(new_circs, filename)
            old_circ_f = os.path.join(old_circs, name)

            with open(new_circ_f, 'r') as f:
                new_circ = Circuit.from_dict(json.load(f))
            with open(old_circ_f, 'r') as f:
                old_circ = Circuit.from_dict(json.load(f))
            print(f'Checking equivalence for {name} ({new_circ.n_qubits} qb, {new_circ.n_gates} gates)')
            if new_circ.n_qubits > MAX_QUBITS:
                print("Skip")
                n_skipped += 1
                continue
            start_time = time.time()
            is_eq = test_equivalence(new_circ, old_circ)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if is_eq:
                print(f'{name}: OK ({elapsed_time:.2f}s))')
                n_success += 1
            else:
                print(f'{name}: FAIL ({elapsed_time:.2f}s))')
                n_fail += 1
    print(f"Done. Success/Fail/Skipped ({n_success}/{n_fail}/{n_skipped}).")
