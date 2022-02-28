import numpy as np
import qiskit.quantum_info as qi

def get_PBasis(n_qubits):
    # return list of Pauli basis elements as np.array
    PL = qi.pauli_basis(n_qubits, pauli_list = True)
    PBasis = [1/np.sqrt(2**n_qubits)* qi.Pauli(s).to_matrix() for s in PL]
    return PBasis


def superket(rho):
    # takes a density matrix [np.array] as input and returns its vectorization in
    # the Hilbert Schmidt space according to the num_qubit Pauli basis [list of np.array].
    n = len(rho[0])
    n_q = int(np.log2(n))
    PBasis = get_PBasis(n_q)
    d = n**2
    vec_rho = np.zeros(shape=(d,1))
    for i in range(d):
        vec_rho[i] = np.real(np.trace(rho.dot(PBasis[i]))) # Note: coefficients are real if we use Pauli (hermitian) basis

    return vec_rho

def superbra(rho):
    # takes a density matrix [np.array] as input and returns its vectorization in
    # the Hilbert Schmidt dual space according to the num_qubit Pauli basis [list of np.array].
    n = len(rho[0])
    n_q = int(np.log2(n))
    PBasis = get_PBasis(n_q)
    d = n**2
    vec_rho = np.zeros(shape=(d,1))
    for i in range(d):
        vec_rho[i] = np.real(np.trace(rho.dot(PBasis[i]))) # Note: coefficients are real if we use Pauli (hermitian) basis

    return vec_rho.conjugate().T


def gate_to_channelmatrix(G):
    # takes a np.array gate G and return a a np.array channel matrix of G
    n = len(G[0])
    n_q = int(np.log2(n))
    PBasis = get_PBasis(n_q)
    d = n**2
    G_dag = G.conjugate().T
    cm = np.zeros(shape=(d,d))
    for a in range(d):
        for b in range(d):
            cm[a,b] = np.real(np.trace( PBasis[a]@G@PBasis[b]@G_dag))
            #cm[a,b] = cm[a,b] if np.abs(cm[a,b]) > 1e-15 else 0 # real coef since we're using Pauli Basis
    return cm

def my_braket(L):
    # takes as input a list and return the "curcuit" result of these operations
    C = np.eye(L[1].shape[0])
    if len(L) > 2:
        for i in range(1,len(L)-1):
            C = np.dot(C,L[i])

    if L[0].shape[1] == 1:
        L[0] = L[0].T

    if not np.equal(C,np.eye(L[1].shape[0])).all():
        right = np.dot(C,L[-1])
    else:
        right = L[-1]
    return np.dot(L[0],right)



# Metrics


def AGsI(gate_set_tm, noisy_gate_set_tm):
    Gamma = len(gate_set_tm)
    AGsI = 0
    d = np.sqrt(gate_set_tm[0].shape[0])
    for gamma in range(Gamma):
        gate = gate_set_tm[gamma].conjugate().T
        gate_noisy = noisy_gate_set_tm[gamma]
        Fid = (np.trace(gate@gate_noisy)+d)/(d*(d+1))
        AGsI += 1-Fid
    AGsI = AGsI/Gamma
    return AGsI

def statistical_distance(p1,p2):
    return(0.5 * np.linalg.norm((p2-p1),ord=1))



# Noise models (single qubit)
def get_AD_channel(gamma):
    d_vec = np.array([1,np.sqrt(1-gamma),np.sqrt(1-gamma),1-gamma])
    AD = np.diag(d_vec)
    AD[3,0] = gamma
    return AD

def get_AD_channel2(gamma):
    M0 = np.array([[1, 0],[0, np.sqrt(1-gamma)]])
    M1 = np.array([[0, np.sqrt(gamma)],[0,0]])
    n = len(M0[0])
    n_q = int(np.log2(n))
    PBasis = get_PBasis(n_q)
    d = n**2
    cm = np.zeros(shape=(d,d))
    for a in range(d):
        for b in range(d):
            cm[a,b] = np.real(np.trace( PBasis[a]@ ( M1@PBasis[b]@M1 + M0@PBasis[b]@M0.T ) ) ) # real coef since we're using Pauli Basis
    return cm

def get_Pauli_channel(px,py,pz):
    d_vec = np.array([1,1-2*(py+pz),1-2*(px+pz),1-2*(px+py)])
    PC = np.diag(d_vec)
    return PC


def get_Pauli_channel2(px,py,pz):
    X = qi.Pauli('X').to_matrix()
    Y = qi.Pauli('Y').to_matrix()
    Z = qi.Pauli('Z').to_matrix()

    n = 2
    n_q = 1
    PBasis = get_PBasis(n_q)
    d = n**2
    cm = np.zeros(shape=(d,d))
    for a in range(d):
        for b in range(d):
            cm[a,b] = np.real(np.trace( PBasis[a]@( (1-px-py-pz)*PBasis[b] + px*X@PBasis[b]@X + py*Y@PBasis[b]@Y + pz*Z@PBasis[b]@Z ) ) )

    return cm

def get_rotation_channel(theta,axis):
    return(gate_to_channelmatrix( get_rotation_gate(theta,axis) ))

def get_rotation_gate(theta,axis):
    # axis is a string defining the axis of rotation on the bloch sphere
    if axis not in ['x','y','z']:
        print('Invalid axis')
        return 0
    R = np.cos(theta/2)*np.eye(2) - 1.j * np.sin(theta/2)*qi.Pauli(axis.upper()).to_matrix()
    return(R)
