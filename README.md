# Secure Quantum Handshakes: Quantum Network Verification Via Simon's Algorithm

This repository contains an implementation of a **pedagogical approximation** of a quantum network admission protocol using classical communication.
A client uses a quantum computer to query a Simon oracle and returns the recovered hidden string to a server, which checks the answer and the response time.
Because the client receives the oracle implementation, a classical attacker may recover the hidden string without quantum queries; this approximation **does not verify quantum network integrity**.
See the associated paper for the full protocol description: [paper link].

## Table of Contents

- [Introduction](#introduction)
- [Structure](#structure)
- [Usage](#usage)
- [License](#license)

## Introduction

This repository implements a classical-communication variant of a quantum network admission protocol, designed as a pedagogical tool for exploring how the secure protocol works.
In the variant, a prospective participant uses a quantum computer to query a Simon oracle, recover its hidden string, and return that string to the network operator.
The operator admits the participant only if the answer is correct and arrives within a specified time limit.

Because the oracle's implementation must be shared with the participant, a classical attacker could potentially recover the hidden string without making quantum queries.
The protocol therefore does not verify quantum network integrity; it instead offers a practical way to build intuition about the secure protocol.
Implementations for IBM quantum devices and simulators are provided.

## Structure

The repository structure is as follows:

```
adiabatic-simons-algorithm/
├── simon_handshake.py                # Executes the classical version of the network verification file
├── test_oracle.py                    # Tests the oracle function
├── noisySimulations/                 # Contains code and results for noisy simulations of the ideal protocol
├── figures/                          # Contains figures for the paper
├── archive/                          # Old code, data, and figure files
├── .gitignore                        # Git ignore file (e.g., system files)
├── LICENSE                           # License file
└── README.md                         # Project description and instructions
```

Of these files, `simon_handshake.py` is the most important, as it allows for the execution of the classical approximation to the quantum network verification protocol.
Following this, the files in the `noisySimulations` directory will be useful to those looking to reproduce the results of our paper.

## Usage

To begin, clone the repository.
To run the approximate network verification protocol, first set the parameters at the top of the `simon_handshake.py` file.
Next, start two parallel terminal instances; execute `python simon_handshake.py alice` in the first, and `python simon_handshake.py bob` in the second.
The terminals will then record the logs as Alice (the server) and Bob (the client) complete the handshake.

## License

This project is licensed under the MIT License — see the LICENSE file for details.
