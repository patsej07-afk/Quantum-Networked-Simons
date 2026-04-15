# simon_handshake_ibm.py
# Run:  python simon_handshake_ibm.py alice
#       python simon_handshake_ibm.py bob

# -------------------- imports --------------------

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_aer.aerprovider import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
import socket, json, secrets, hmac, hashlib, io, struct
from qiskit.qpy import dump as qpy_dump, load as qpy_load
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit_ibm_runtime import QiskitRuntimeService, fake_provider, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# -------------------- networking constants --------------------

HOST = '127.0.0.1'
PORT = 5055

# -------------------- networking helpers --------------------

def send_frame(conn: socket.socket, payload: bytes):
    '''
    Send a length-prefixed message over the socket.
    Args:
        conn (socket.socket): the socket connection to send on.
        payload (bytes): the bytes to send as the message body.
    '''

    conn.sendall(struct.pack('!I', len(payload)) + payload)

def recv_frame(conn: socket.socket) -> bytes:
    '''
    Receive a length-prefixed message from the socket.
    Args:
        conn (socket.socket): the socket connection to receive from.
    Returns:
        The payload bytes.
    '''

    hdr = conn.recv(4)
    if len(hdr) < 4:
        raise ConnectionError('Disconnected')
    (n,) = struct.unpack('!I', hdr)
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('Disconnected')
        buf += chunk
    return buf

def send_json(conn: socket.socket, obj: dict):
    '''
    Send a JSON-serializable object as a length-prefixed message.
    Args:
        conn (socket.socket): the socket connection to send on.
        obj (dict): the JSON-serializable object to send (usually a dict).
    '''

    send_frame(conn, json.dumps(obj).encode())

def recv_json(conn: socket.socket) -> dict:
    '''
    Receive a length-prefixed JSON message and parse it.
    Args:
        conn (socket.socket): the socket connection to receive from.
    Returns:
        The parsed JSON object (usually a dict).
    '''

    return json.loads(recv_frame(conn).decode())

def send_qpy_circuit(conn: socket.socket, qc: QuantumCircuit):
    '''
    Serialize a QuantumCircuit and send it as a length-prefixed message.
    Args:
        conn (socket.socket): the socket connection to send on.
        qc (QuantumCircuit): the circuit to serialize and send.
    '''

    bio = io.BytesIO()
    qpy_dump(qc, bio)
    send_frame(conn, bio.getvalue())

def recv_qpy_circuit(conn: socket.socket) -> QuantumCircuit:
    '''
    Receive a length-prefixed message of a serialized QuantumCircuit and parse.
    Args:
        conn (socket.socket): the socket connection to receive from.
    Returns:
        The deserialized QuantumCircuit.
    '''

    bio = io.BytesIO(recv_frame(conn))
    (qc,) = qpy_load(bio)
    return qc

# -------------------- GF(2) linear algebra --------------------

def gf2_rref(A: np.ndarray) -> np.ndarray | list:
    '''
    Compute the RREF of A over GF(2) and return the pivot columns.
    A is modified in-place but also returned for convenience.
    Args:
        A (np.ndarray): m-by-n binary matrix (dtype=np.uint8).
    Returns:
        R (np.ndarray): the RREF of A.
        pivots (list of int): the list of pivot column indices.
    '''

    A = (A.copy() % 2).astype(np.uint8)
    m, n = A.shape
    r = 0
    pivots = []
    for c in range(n):
        if r >= m:
            break
        pivot = None
        for rr in range(r, m):
            if A[rr, c] == 1:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        pivots.append(c)
        for rr in range(m):
            if rr != r and A[rr, c] == 1:
                A[rr] ^= A[r]
        r += 1
    return A, pivots

def gf2_nullspace_vector(A: np.ndarray) -> np.ndarray:
    '''
    Return one nonzero vector in nullspace(A) over GF(2), or zeros if not found.
    Args:
        A (np.ndarray): m-by-n binary matrix (dtype=np.uint8).
    Returns:
        v (np.ndarray): a nonzero vector such that Av=0 mod 2, else zeros.
    '''

    R, pivots = gf2_rref(A)
    m, n = R.shape
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]
    for free in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[free] = 1
        pivot_row = 0
        for p in pivots:
            s = 0
            for j in range(n):
                if j != p and R[pivot_row, j] == 1 and v[j] == 1:
                    s ^= 1
            v[p] = s
            pivot_row += 1
        if np.any(v == 1):
            return v
    return np.zeros(n, dtype=np.uint8)

# -------------------- Simon oracle builder --------------------
# Build a valid Simon oracle U_f such that f(x)=f(x⊕s) (2-to-1 with period s).
# We use: always apply y ^= Mx, then (controlled on x0) apply y ^= M s.

def random_invertible_matrix(n: int) -> np.ndarray:
    '''
    Generate a random n-by-n invertible matrix over GF(2).
    Args:
        n (int): the number of rows and columns.
    Returns:
        M (np.ndarray): a random n-by-n invertible binary matrix.
    '''

    while True:
        M = np.random.randint(0, 2, size=(n, n), dtype=np.uint8)
        _, pivots = gf2_rref(M)
        if len(pivots) == n:
            return M

def apply_linear_map(qc: QuantumCircuit, x: list, y: list, M: np.ndarray):
    '''
    Apply the linear map defined by M as CNOTs from x to y.
    Args:
        qc (QuantumCircuit): the circuit to apply the map on.
        x (list of int): the list of qubit indices for the input register.
        y (list of int): the list of qubit indices for the output register.
        M (np.ndarray): the n-by-n binary matrix defining the linear map.
    '''

    n = len(x)
    for row in range(n):
        for col in range(n):
            if M[row, col] == 1:
                qc.cx(x[col], y[row])

def build_simon_oracle(n: int, s_bits: list) -> QuantumCircuit:
    '''
    Build a valid Simon oracle U_f such that f(x)=f(x⊕s) (2-to-1 with period s).
    Args:
        n (int): number of bits (and qubits) for x and y.
        s_bits (list of int): list of n bits representing the secret string s.
    Returns:
        QuantumCircuit: implemention of U_f on 2n qubits.
    TODO: Could improve this function by applying a random permutation.
    '''

    s = np.array(s_bits, dtype=np.uint8)
    if not np.any(s == 1):
        raise ValueError('Simon requires nonzero s')
    qc = QuantumCircuit(2 * n, name='U_f')
    for i in range(n):
        qc.cx(i, i+n)
    control_bit = s_bits.index(1)
    for j in range(n):
        if s[j] == 1:
            qc.cx(control_bit, j+n)
    return qc

def build_simon_circuit(n: int, oracle: QuantumCircuit) -> QuantumCircuit:
    '''
    Build the full Simon circuit to recover s given oracle U_f.
    Args:
        n (int): number of bits/qubits for x and y.
        oracle (QuantumCircuit): the circuit implementing U_f on 2n qubits.
    Returns:
        QuantumCircuit: the full Simon circuit with n measurement bits.
    '''

    qc = QuantumCircuit(2 * n, n)
    qc.h(range(n))
    qc.compose(oracle, inplace=True)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc

# -------------------- key + auth helpers --------------------

def kdf_session_key(s_bits: bytes, nonce_a: bytes, nonce_b: bytes) -> bytes:
    '''
    Derive a session key from the secret string and nonces.
    Args:
        s_bits (bytes): the secret string as bytes.
        nonce_a (bytes): the nonce from Alice.
        nonce_b (bytes): the nonce from Bob.
    Returns:
        bytes: the derived session key.
    '''

    return hashlib.sha256(
        b'SIMON-HS|' + s_bits + b'|' + nonce_a + b'|' + nonce_b
    ).digest()

def hmac_proof(key: bytes, transcript: bytes) -> str:
    '''
    Compute an HMAC proof of the transcript using the session key.
    Args:
        key (bytes): the session key to use for HMAC.
        transcript (bytes): the transcript of the handshake to authenticate.
    Returns:
        str: the hex-encoded HMAC digest as the proof.
    '''

    return hmac.new(key, transcript, hashlib.sha256).hexdigest()

# -------------------- IBM run helper --------------------

def get_ibm_backend_and_pm(
        backend_name: str,
        optimization_level: int = 1
    ) -> Backend | StagedPassManager:
    '''
    Get the IBM backend and a corresponding pass manager for optimization.
    Can be a current IBM device, the fake Brisbane sim., or a noiseless sim.
    Reads the API token from './APIs/IBM_API.txt' and initializes the service.
    Args:
        backend_name (str): the name of the IBM backend to use.
        optimization_level (int): the level of optimization to apply.
    Returns:
        Backend | StagedPassManager: the IBM backend and preset pass manager.
    '''

    with open('./APIs/IBM_API.txt', 'r') as file:
        token = file.read()
    service = QiskitRuntimeService(channel='ibm_cloud', token=token)
    if backend_name == 'ibm_fake_brisbane':
        backend = fake_provider.FakeBrisbane()
    elif backend_name == 'ibm_noiseless_simulator':
        backend = AerSimulator()
    else:
        backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level
    )
    return backend, pm

def run_on_ibm_sampler(
        qc: QuantumCircuit,
        backend: Backend,
        pm: StagedPassManager,
        shots: int = 1024
    ) -> dict:
    '''
    Transpile the circuit with the given pass manager and run it on the IBM sampler.
    Args:
        qc (QuantumCircuit): the circuit to run.
        backend (Backend): the IBM backend to run on.
        pm (StagedPassManager): the pass manager for transpilation.
        shots (int): the number of shots to execute.
    Returns:
        dict: the counts result from the sampler.
    '''
    isa_circuit = pm.run(qc)
    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        job = sampler.run([isa_circuit], shots=shots)
        res = job.result()[0]
        return res.data.c.get_counts()

# -------------------- roles --------------------

def alice(ns: int|list, iters: int):
    '''
    Perform the Alice role of the Simon handshake protocol.
    Args:
        ns (int|list): the n values to test (number of bits/qubits).
        iters (int): the number of iterations to run for each n.
    '''

    successesDict = dict()
    if type(ns) == int:
        ns = [ns]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind((HOST, PORT))
        srv.listen(1)
        print(f'[Alice] listening on {HOST}:{PORT}')
        conn, _ = srv.accept()

        with conn:
            for i in range(1, iters + 1):
                print(f'[Alice] Start iteration {i}')
                for n in ns:
                    print(f'[Alice] Begin n       = {n}')
                    # 1) choose secret s and nonceA
                    s = [1] * n         # Use the string of all 1s
                    nonce_a = secrets.token_bytes(16)

                    oracle = build_simon_oracle(n, s)
                    # 2) hello
                    send_json(conn, {'type': 'hello', 'proto': 'simon-hs', 'n': n, 'nonce_a': nonce_a.hex()})
                    msg = recv_json(conn)
                    if msg.get('type') != 'hello_ack':
                        raise RuntimeError('bad ack')
                    nonce_b = bytes.fromhex(msg['nonce_b'])
                    backend_b = msg['backend']

                    # 3) send oracle circuit (QPY)
                    send_json(conn, {'type': 'oracle_qpy'})
                    send_qpy_circuit(conn, oracle)

                    # 4) receive proof from Bob
                    proof_msg = recv_json(conn)
                    if proof_msg.get('type') != 'proof':
                        raise RuntimeError('missing proof')
                    proof_hex = proof_msg['hmac']

                    # 5) derive session key (Alice knows s already)
                    s_bytes = bytes(int(''.join(map(str, s)), 2).to_bytes((n + 7)//8, 'big'))
                    session_key = kdf_session_key(s_bytes, nonce_a, nonce_b)

                    transcript = (f'n={n}|nonce_a={nonce_a.hex()}|nonce_b={nonce_b.hex()}|backend={backend_b}').encode()
                    expected = hmac_proof(session_key, transcript)
                    isSuccess = hmac.compare_digest(expected, proof_hex)

                    print(f'[Alice] secret s      = {s}')
                    # print(f'[Alice] session key   = {session_key.hex()[:32]}...')
                    print(f'[Alice] proof valid?  = {isSuccess}')
                    if n not in successesDict.keys():
                        successesDict[n] = int(isSuccess)
                    else:
                        successesDict[n] += int(isSuccess)

    print('[Alice] Connection disconnected')
    normalizedSuccessesDict = {k: v/iters for k, v in successesDict.items()}
    with open(
        f'./data/{BACKEND_NAME}-{str(NS)}-{ITERS}-{SHOTS}-{OPTIMIZATION_LEVEL}',
        'w+'
    ) as file:
        file.write(str(normalizedSuccessesDict))

def bob(backend_name: str, optimization_level: int, shots: int):
    '''
    Perform the Bob role of the Simon handshake protocol.
    Args:
        backend_name (str): the name of the IBM backend to use.
        optimization_level (int): the level of optimization for transpilation.
        shots (int): the number of shots to execute for each run (n + shots).
    '''

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sck:
        sck.connect((HOST, PORT))

        backend, pm = get_ibm_backend_and_pm(backend_name, optimization_level=optimization_level)

        while True:
            try:
                hello = recv_json(sck)
            except ConnectionError:
                print('[Bob] Connection disconnected')
                break
            if hello.get('type') != 'hello':
                raise RuntimeError('bad hello')
            n = int(hello['n'])
            nonce_a = bytes.fromhex(hello['nonce_a'])
            nonce_b = secrets.token_bytes(16)

            send_json(sck, {'type': 'hello_ack', 'nonce_b': nonce_b.hex(), 'backend': backend_name})

            # oracle incoming
            msg = recv_json(sck)
            if msg.get('type') != 'oracle_qpy':
                raise RuntimeError('expected oracle')
            oracle = recv_qpy_circuit(sck)

            # Build Simon circuit and run on IBM backend
            qc = build_simon_circuit(n, oracle)
            counts = run_on_ibm_sampler(qc, backend, pm, shots=n+shots)
            sorted_counts = dict(sorted(counts.items(), key=lambda item: -item[1]))

            # Extract independent y samples (constraints y·s=0)
            samples = []
            for bitstr, ct in sorted_counts.items():
                y = np.array([int(b) for b in bitstr[::-1]], dtype=np.uint8)  # endian fix
                if np.any(y == 1):
                    samples.append(y)
                if len(samples) >= 2 * n:  # usually enough
                    break
            A = np.vstack(samples) if samples else np.zeros((0, n), dtype=np.uint8)
            s_hat = gf2_nullspace_vector(A)

            # derive same session key from recovered s_hat
            s_int = int(''.join(str(int(b)) for b in s_hat.tolist()), 2)
            s_bytes = s_int.to_bytes((n + 7)//8, 'big')
            session_key = kdf_session_key(s_bytes, nonce_a, nonce_b)

            transcript = (f'n={n}|nonce_a={nonce_a.hex()}|nonce_b={nonce_b.hex()}|backend={backend_name}').encode()
            proof = hmac_proof(session_key, transcript)

            print(f'[Bob] recovered s_hat = {s_hat.tolist()}')
            # print(f'[Bob] session key     = {session_key.hex()[:32]}...')

            send_json(sck, {'type': 'proof', 'hmac': proof})

# -------------------- main --------------------

# BACKEND_NAME = 'ibm_fake_brisbane'
BACKEND_NAME = 'ibm_noiseless_simulator'
# BACKEND_NAME = 'ibm_kingstone'
# BACKEND_NAME = 'ibm_fez'
# BACKEND_NAME = 'ibm_marrakesh'
NS = [2, 4, 6]
ITERS = 1
SHOTS = 20
OPTIMIZATION_LEVEL = 1

if __name__ == '__main__':
    import sys
    role = sys.argv[1] if len(sys.argv) > 1 else 'alice'
    if role == 'alice':
        alice(NS, ITERS)
    else:
        bob(BACKEND_NAME, OPTIMIZATION_LEVEL, SHOTS)


# def alice(n=2):
#     # 1) choose secret s and nonceA
#     # s = [secrets.randbelow(2) for _ in range(n)]
#     s = [1] * n         # Use the string of all 1s
#     while all(b == 0 for b in s):
#         s = [secrets.randbelow(2) for _ in range(n)]
#     nonce_a = secrets.token_bytes(16)

#     oracle = build_simon_oracle(n, s)

#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
#         srv.bind((HOST, PORT))
#         srv.listen(1)
#         print(f'[Alice] listening on {HOST}:{PORT}')
#         conn, _ = srv.accept()
#         with conn:
#             # 2) hello
#             send_json(conn, {'type': 'hello', 'proto': 'simon-hs', 'n': n, 'nonce_a': nonce_a.hex()})
#             msg = recv_json(conn)
#             if msg.get('type') != 'hello_ack':
#                 raise RuntimeError('bad ack')
#             nonce_b = bytes.fromhex(msg['nonce_b'])
#             backend_b = msg['backend']

#             # 3) send oracle circuit (QPY)
#             send_json(conn, {'type': 'oracle_qpy'})
#             send_qpy_circuit(conn, oracle)

#             # 4) receive proof from Bob
#             proof_msg = recv_json(conn)
#             if proof_msg.get('type') != 'proof':
#                 raise RuntimeError('missing proof')
#             proof_hex = proof_msg['hmac']

#             # 5) derive session key (Alice knows s already)
#             s_bytes = bytes(int(''.join(map(str, s)), 2).to_bytes((n + 7)//8, 'big'))
#             session_key = kdf_session_key(s_bytes, nonce_a, nonce_b)

#             transcript = (f'n={n}|nonce_a={nonce_a.hex()}|nonce_b={nonce_b.hex()}|backend={backend_b}').encode()
#             expected = hmac_proof(session_key, transcript)

#             print(f'[Alice] secret s       = {s}')
#             print(f'[Alice] session key   = {session_key.hex()[:32]}...')
#             print(f'[Alice] proof valid?  = {hmac.compare_digest(expected, proof_hex)}')

# def bob(backend_name='ibm_brisbane', shots=2048):
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sck:
#         sck.connect((HOST, PORT))

#         hello = recv_json(sck)
#         if hello.get('type') != 'hello':
#             raise RuntimeError('bad hello')
#         n = int(hello['n'])
#         nonce_a = bytes.fromhex(hello['nonce_a'])
#         nonce_b = secrets.token_bytes(16)

#         send_json(sck, {'type': 'hello_ack', 'nonce_b': nonce_b.hex(), 'backend': backend_name})

#         # oracle incoming
#         msg = recv_json(sck)
#         if msg.get('type') != 'oracle_qpy':
#             raise RuntimeError('expected oracle')
#         oracle = recv_qpy_circuit(sck)

#         # Build Simon circuit and run on IBM backend
#         qc = build_simon_circuit(n, oracle)
#         counts = run_on_ibm_sampler(qc, backend_name=backend_name, shots=shots)
#         sorted_counts = dict(sorted(counts.items(), key=lambda item: -item[1]))

#         # Extract independent y samples (constraints y·s=0)
#         samples = []
#         for bitstr, ct in sorted_counts.items():
#             y = np.array([int(b) for b in bitstr[::-1]], dtype=np.uint8)  # endian fix
#             if np.any(y == 1):
#                 samples.append(y)
#             if len(samples) >= 2 * n:  # usually enough
#                 break
#         A = np.vstack(samples) if samples else np.zeros((0, n), dtype=np.uint8)
#         s_hat = gf2_nullspace_vector(A)

#         # derive same session key from recovered s_hat
#         s_int = int(''.join(str(int(b)) for b in s_hat.tolist()), 2)
#         s_bytes = s_int.to_bytes((n + 7)//8, 'big')
#         session_key = kdf_session_key(s_bytes, nonce_a, nonce_b)

#         transcript = (f'n={n}|nonce_a={nonce_a.hex()}|nonce_b={nonce_b.hex()}|backend={backend_name}').encode()
#         proof = hmac_proof(session_key, transcript)

#         print(f'[Bob] recovered s_hat = {s_hat.tolist()}')
#         print(f'[Bob] session key     = {session_key.hex()[:32]}...')

#        send_json(sck, {'type': 'proof', 'hmac': proof})