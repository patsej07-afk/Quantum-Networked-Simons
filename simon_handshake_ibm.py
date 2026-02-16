# simon_handshake_ibm.py
# Run:  python simon_handshake_ibm.py alice
#       python simon_handshake_ibm.py bob

import socket, json, secrets, hmac, hashlib, io, struct
import numpy as np

from qiskit import QuantumCircuit
from qiskit.qpy import dump as qpy_dump, load as qpy_load

from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler  # :contentReference[oaicite:3]{index=3}

HOST = "127.0.0.1"
PORT = 5055

# -------------------- networking helpers --------------------

def send_frame(conn, payload: bytes):
    conn.sendall(struct.pack("!I", len(payload)) + payload)

def recv_frame(conn) -> bytes:
    hdr = conn.recv(4)
    if len(hdr) < 4:
        raise ConnectionError("Disconnected")
    (n,) = struct.unpack("!I", hdr)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Disconnected")
        buf += chunk
    return buf

def send_json(conn, obj):
    send_frame(conn, json.dumps(obj).encode())

def recv_json(conn):
    return json.loads(recv_frame(conn).decode())

def send_qpy_circuit(conn, qc: QuantumCircuit):
    bio = io.BytesIO()
    qpy_dump(qc, bio)
    send_frame(conn, bio.getvalue())

def recv_qpy_circuit(conn) -> QuantumCircuit:
    bio = io.BytesIO(recv_frame(conn))
    (qc,) = qpy_load(bio)
    return qc

# -------------------- GF(2) linear algebra --------------------

def gf2_rref(A):
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

def gf2_nullspace_vector(A):
    """Return one nonzero vector in nullspace(A) over GF(2), or zeros if not found."""
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
# We use: always apply y ^= Mx, then (controlled on x0) apply y ^= M s

def random_invertible_matrix(n):
    while True:
        M = np.random.randint(0, 2, size=(n, n), dtype=np.uint8)
        _, pivots = gf2_rref(M)
        if len(pivots) == n:
            return M

def apply_linear_map(qc, x, y, M):
    n = len(x)
    for row in range(n):
        for col in range(n):
            if M[row, col] == 1:
                qc.cx(x[col], y[row])

def build_simon_oracle(n, s_bits):
    s = np.array(s_bits, dtype=np.uint8)
    if not np.any(s == 1):
        raise ValueError("Simon requires nonzero s")
    M = random_invertible_matrix(n)

    qc = QuantumCircuit(2 * n, name="U_f")
    x = list(range(n))
    y = list(range(n, 2 * n))

    apply_linear_map(qc, x, y, M)               # y ^= Mx
    Ms = (M @ s) % 2
    ctrl = x[0]
    for i in range(n):
        if Ms[i] == 1:
            qc.cx(ctrl, y[i])                   # if x0==1: y ^= Ms
    return qc

def build_simon_circuit(n, oracle: QuantumCircuit):
    qc = QuantumCircuit(2 * n, n)
    qc.h(range(n))
    qc.compose(oracle, inplace=True)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc

# -------------------- key + auth helpers --------------------

def kdf_session_key(s_bits: bytes, nonce_a: bytes, nonce_b: bytes) -> bytes:
    # simple KDF; swap for HKDF if you want
    return hashlib.sha256(b"SIMON-HS|" + s_bits + b"|" + nonce_a + b"|" + nonce_b).digest()

def hmac_proof(key: bytes, transcript: bytes) -> str:
    return hmac.new(key, transcript, hashlib.sha256).hexdigest()

# -------------------- IBM run helper --------------------

def run_on_ibm_sampler(qc: QuantumCircuit, backend_name: str, shots: int = 1024):
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    with Session(service=service, backend=backend) as session:  # :contentReference[oaicite:4]{index=4}
        sampler = Sampler(mode=session)                         # :contentReference[oaicite:5]{index=5}
        job = sampler.run([qc], shots=shots)
        res = job.result()[0]
        # Return counts as {bitstring: count}
        return res.data.cr.get_counts()

# -------------------- roles --------------------

def alice(n=5, backend_name="ibm_brisbane"):
    # 1) choose secret s and nonceA
    s = [secrets.randbelow(2) for _ in range(n)]
    while all(b == 0 for b in s):
        s = [secrets.randbelow(2) for _ in range(n)]
    nonce_a = secrets.token_bytes(16)

    oracle = build_simon_oracle(n, s)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind((HOST, PORT))
        srv.listen(1)
        print(f"[Alice] listening on {HOST}:{PORT}")
        conn, _ = srv.accept()
        with conn:
            # 2) hello
            send_json(conn, {"type": "hello", "proto": "simon-hs", "n": n, "nonce_a": nonce_a.hex()})
            msg = recv_json(conn)
            if msg.get("type") != "hello_ack":
                raise RuntimeError("bad ack")
            nonce_b = bytes.fromhex(msg["nonce_b"])
            backend_b = msg["backend"]

            # 3) send oracle circuit (QPY)
            send_json(conn, {"type": "oracle_qpy"})
            send_qpy_circuit(conn, oracle)

            # 4) receive proof from Bob
            proof_msg = recv_json(conn)
            if proof_msg.get("type") != "proof":
                raise RuntimeError("missing proof")
            proof_hex = proof_msg["hmac"]

            # 5) derive session key (Alice knows s already)
            s_bytes = bytes(int("".join(map(str, s)), 2).to_bytes((n + 7)//8, "big"))
            session_key = kdf_session_key(s_bytes, nonce_a, nonce_b)

            transcript = (f"n={n}|nonce_a={nonce_a.hex()}|nonce_b={nonce_b.hex()}|backend={backend_b}").encode()
            expected = hmac_proof(session_key, transcript)

            print(f"[Alice] secret s       = {s}")
            print(f"[Alice] session key   = {session_key.hex()[:32]}...")
            print(f"[Alice] proof valid?  = {hmac.compare_digest(expected, proof_hex)}")

def bob(backend_name="ibm_brisbane", shots=2048):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sck:
        sck.connect((HOST, PORT))

        hello = recv_json(sck)
        if hello.get("type") != "hello":
            raise RuntimeError("bad hello")
        n = int(hello["n"])
        nonce_a = bytes.fromhex(hello["nonce_a"])
        nonce_b = secrets.token_bytes(16)

        send_json(sck, {"type": "hello_ack", "nonce_b": nonce_b.hex(), "backend": backend_name})

        # oracle incoming
        msg = recv_json(sck)
        if msg.get("type") != "oracle_qpy":
            raise RuntimeError("expected oracle")
        oracle = recv_qpy_circuit(sck)

        # Build Simon circuit and run on IBM backend
        qc = build_simon_circuit(n, oracle)
        counts = run_on_ibm_sampler(qc, backend_name=backend_name, shots=shots)

        # Extract independent y samples (constraints y·s=0)
        samples = []
        for bitstr, ct in counts.items():
            y = np.array([int(b) for b in bitstr[::-1]], dtype=np.uint8)  # endian fix
            if np.any(y == 1):
                samples.append(y)
            if len(samples) >= 2 * n:  # usually enough
                break
        A = np.vstack(samples) if samples else np.zeros((0, n), dtype=np.uint8)
        s_hat = gf2_nullspace_vector(A)

        # derive same session key from recovered s_hat
        s_int = int("".join(str(int(b)) for b in s_hat.tolist()), 2)
        s_bytes = s_int.to_bytes((n + 7)//8, "big")
        session_key = kdf_session_key(s_bytes, nonce_a, nonce_b)

        transcript = (f"n={n}|nonce_a={nonce_a.hex()}|nonce_b={nonce_b.hex()}|backend={backend_name}").encode()
        proof = hmac_proof(session_key, transcript)

        print(f"[Bob] recovered s_hat = {s_hat.tolist()}")
        print(f"[Bob] session key     = {session_key.hex()[:32]}...")

        send_json(sck, {"type": "proof", "hmac": proof})

if __name__ == "__main__":
    import sys
    role = sys.argv[1] if len(sys.argv) > 1 else "alice"
    if role == "alice":
        alice()
    else:
        # optionally: python simon_handshake_ibm.py bob ibm_kyiv
        backend = sys.argv[2] if len(sys.argv) > 2 else "ibm_brisbane"
        bob(backend_name=backend)


