import pytest
from simon_handshake_ibm import *

def test_build_simon_oracle_all_1():
    n = 5
    s = [1, 1, 1, 1, 1]
    oracle = build_simon_oracle(n, s)

    qc = QuantumCircuit(10)
    for i in range(n):
        qc.cx(i, i+n)
    for j in range(n):
            qc.cx(0, j+n)

    assert oracle == qc

def test_build_simon_oracle_start_with_0():
    n = 5
    s = [0, 1, 1, 1, 1]
    oracle = build_simon_oracle(n, s)

    qc = QuantumCircuit(10)
    for i in range(n):
        qc.cx(i, i+n)
    for j in range(1, n):
            qc.cx(1, j+n)

    assert oracle == qc

def test_build_simon_oracle_all_0():
    n = 5
    s = [0, 0, 0, 0, 0]

    with pytest.raises(ValueError) as e:
        build_simon_oracle(n, s)