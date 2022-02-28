import numpy as np

from src import quantum_utilities as qu

# Import Qiskit basics
from qiskit import QuantumCircuit
from qiskit import Aer, transpile, execute
from qiskit.tools.visualization import plot_histogram

# Import from QiskitAer noise module
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error, amplitude_damping_error, coherent_unitary_error
from qiskit.providers.aer.noise import depolarizing_error

# Import Quantum Information module
import qiskit.quantum_info as qi


def build_error_map(e):

    AD = amplitude_damping_error(e[0])
    Pauli = pauli_error([('X', e[1]), ('Y', e[2]), ('Z', e[3]), ('I', 1 - sum(e[1:4]))])
    Rx = np.array([[np.cos(e[4]/2), -np.sin(e[4]/2)*1j],[-np.sin(e[4]/2)*1j, np.cos(e[4]/2)]],dtype = np.complex_)
    Ry = np.array([[np.cos(e[5]/2), -np.sin(e[5]/2)],[np.sin(e[5]/2), np.cos(e[5]/2)]],dtype = np.complex_)
    Rz = np.array([[np.exp(-e[6]/2*1j), 0],[0, np.exp(e[6]/2*1j)]],dtype = np.complex_)
    Rx_error = coherent_unitary_error(Rx)
    Ry_error = coherent_unitary_error(Ry)
    Rz_error = coherent_unitary_error(Rz)
    error = Rz_error.compose(Ry_error)
    error = error.compose(Rx_error)
    error = error.compose(Pauli)
    error = error.compose(AD)

    return(error)

def build_noise_model(native_gate_list, build_error_map, error_scale):

    noise_model = NoiseModel(native_gate_list)
    gate_error_params = [] # list of error parameters for each gate

    for gate in native_gate_list:
        e = np.random.random_sample(7)*error_scale
        gate_error_params.append(e)
        error = build_error_map(e)
        noise_model.add_all_qubit_quantum_error(error, gate)

    return(noise_model, gate_error_params)


def build_noise_set(native_gate_op, build_error_map, error_scale):
    # native gate op is a list of np.arrays each containing a native gate
    # we map them into channel matrices and add error to them
    noise_gate_set = []
    gate_error_params = [] # list of error parameters for each gate

    for gate in native_gate_op:
        e = np.random.random_sample(7)*error_scale
        gate_error_params.append(e)
        error = build_error_map(e)
        error_qc = error.to_quantumchannel()
        noise_gate_set.append( error_qc.data @ qu.gate_to_channelmatrix(gate) )

    return(noise_gate_set, gate_error_params)


def sample_circuit(native_gate_list, num_qubits, l):
    circ_id = np.random.randint(len(native_gate_list)-1,size = l)
    gatelist = [qi.Pauli(g.upper()) for g in [native_gate_list[i] for i in circ_id ]]
    qc = QuantumCircuit(num_qubits)
    #operatorlist = [gate.to_matrix() for gate in gatelist]
    for gate in gatelist:
        qc.append(gate.to_instruction(),[0]) # always append to first qubit
    qc.save_density_matrix()
    qc.measure_all()
    return qc, circ_id



def run_experiment(circuit, sim_ideal, sim_noisy, rho_in, P0):

    num_qubits = int(np.log2(len(rho_in[0])))
    noisy_init = QuantumCircuit(num_qubits,1)
    noisy_init.set_density_matrix(rho_in)

    circ_tr_id = transpile(circuit,sim_ideal)
    circ_tr_no = transpile(noisy_init.compose(circuit),sim_noisy)

    result_ideal = sim_ideal.run(circ_tr_id).result()
    result_noisy = sim_noisy.run(circ_tr_no).result()

    # define appropriate measurement operator
    P0_ideal = np.array([[1, 0],[0, 0]], dtype = np.complex_)
    P1_ideal = np.array([[1, 0],[0, 1]], dtype = np.complex_) - P0_ideal
    M_OP_ideal = qi.Operator(0*P0_ideal + 1*P1_ideal)
    M_OP_noisy = qi.Operator(0*P0 + 1*(np.array([[1, 0],[0, 1]],dtype=np.complex_) - P0))


    rho_ideal = result_ideal.data(0)['density_matrix']
    p_ideal = np.real(rho_ideal.expectation_value(M_OP_ideal))
    rho_noisy = result_noisy.data(0)['density_matrix']
    p_noisy = np.real(rho_noisy.expectation_value(M_OP_noisy))


    return p_ideal, p_noisy


def run_test(circuit, noisy_gate_list, rho_in, P0):
    # test is always on 1 qubit stuff (for now)

    # qstate = qu.superket(rho_in)
    # P_mu = qu.superbra(P0)
    qstate = rho_in
    P_mu = P0

    for gate in circuit:
        qstate = noisy_gate_list[gate]@qstate

    p_test = P_mu@qstate

    return p_test



def compute_circuit_coefficients(circ_id, MeasPOVM, rho_in, native_gate_op):
    # circ_id is a list of index of native_gate_op used in the circuit
    # rho_in is the initial state
    # MeasPOVM is a list of measurement POVM effects



    n = len(rho_in[0])
    n_q = int(np.log2(n))
    Basis = qu.get_PBasis(n_q)
    d = n**2
    # Compute C_in,a
    C_ina = []
    C = np.eye(n)
    for gate_id in circ_id:
        C = native_gate_op[gate_id]@C
    for a in Basis:
        C_ina.append( (qu.superbra(a)@qu.gate_to_channelmatrix(C)@qu.superket(rho_in)).item() )

    # Compute C_mu,a
    C_mua_list = []
    for Pmu in MeasPOVM:
        C_mua = []
        for a in Basis:
            C_mua.append( (qu.superbra(Pmu)@qu.gate_to_channelmatrix(C) @ qu.superket(a)).item() )
        C_mua_list.append(C_mua)

    # Compute C_in,mu,gamma,a,b
    C_inmugammaab_list = []
    for Pmu in MeasPOVM:
        C_inmugammaab = []
        for gamma in range(len(native_gate_op)):
            for a in Basis:
                for b in Basis:
                    s = 0
                    for k in range(len(circ_id)):
                        if circ_id[k]==gamma:
                            leftC = np.eye(n)
                            rightC = np.eye(n)
                            for i in range(k):
                                rightC = native_gate_op[circ_id[i]]@rightC
                            for j in range(k,len(circ_id)):
                                leftC = native_gate_op[circ_id[j]]@leftC
                            s += ( qu.superbra(Pmu)@qu.gate_to_channelmatrix(leftC)@qu.superket(a) ).item() * ( qu.superbra(b)@qu.gate_to_channelmatrix(rightC)@qu.superket(rho_in)).item()
                    C_inmugammaab.append(s)
        C_inmugammaab_list.append(C_inmugammaab)

    # Compose matrix rows (one for each Pmu)
    C = np.array( [ [ C_mua_list[0]+C_inmugammaab_list[0]+C_ina ], [ C_mua_list[1]+C_inmugammaab_list[1]+C_ina ] ] , dtype = np.complex_)
    return C


def build_C(total_circuit_id_list, native_gate_op):
    # Define native measurement POVM
    P0_ideal = np.array([[1, 0],[0, 0]], dtype = np.complex_)
    P1_ideal = np.array([[1, 0],[0, 1]], dtype = np.complex_) - P0_ideal
    MeasPOVM = [P0_ideal, P1_ideal]

    # Define ideal initial state
    rho_in_ideal = np.array([[1 , 0],[0, 0]])

    Nc = len(total_circuit_id_list[1])

    d = len(rho_in_ideal[0])**2
    n_cols = d*2 + d**2 * len(native_gate_op)
    n_rows = 2*(len(total_circuit_id_list)-1)*Nc+len(MeasPOVM)
    C = np.zeros(shape = (n_rows, n_cols), dtype = np.complex_)
    for il in range(len(total_circuit_id_list)):
        circ_id_list = total_circuit_id_list[il]
        for ic in range(len(circ_id_list)):
            C_rows = compute_circuit_coefficients(circ_id_list[ic], MeasPOVM, rho_in_ideal, native_gate_op)
            for mu in range(len(MeasPOVM)):
                if(il==0):
                    C[mu] = C_rows[mu] # first length is 0, just measures
                else:
                    C[2+2*((il-1)+ic)+mu] = C_rows[mu]
    return C

def get_noisy_gateset(e, n_q, native_gate_set):
    # Define native measurement POVM
    P0_ideal = np.array([[1, 0],[0, 0]], dtype = np.complex_)
    P1_ideal = np.array([[1, 0],[0, 1]], dtype = np.complex_) - P0_ideal
    MeasPOVM = [P0_ideal, P1_ideal]

    # Define ideal initial state
    rho_in_ideal = np.array([[1 , 0],[0, 0]])

    # split e into its components and compute noisy estimate for gate set
    native_len = len(native_gate_set)
    PBasis = qu.get_PBasis(n_q)
    LB = len(PBasis)

    e_in = e[0:LB]
    est_init = qu.superket(rho_in_ideal) + e_in

    e_gamma_list = [np.zeros(shape = (LB,LB))]*native_len
    est_gate_set = [np.zeros(shape = (LB,LB))]*native_len
    for gamma in range(native_len):
        tmp = e[LB + (gamma)*(LB**2):LB + (gamma+1)*(LB**2)]
        for row in range(LB):
            e_gamma_list[gamma][row] = tmp[row*LB:(row+1)*LB].reshape(tmp[row*LB:(row+1)*LB].shape[0]) ######
        est_gate_set[gamma] = (np.eye(LB) + e_gamma_list[gamma])@qu.gate_to_channelmatrix(native_gate_set[gamma])

    e_ro = e[(LB + (native_len)*(LB**2)):]
    est_measure = qu.superbra(MeasPOVM[0]) + e_ro

    return est_init, est_gate_set, est_measure




z0_matrix = qu.superket(np.array([[1,0],[0,0]],dtype = np.complex_))@qu.superbra(np.array([[1,0],[0,0]],dtype = np.complex_))
z0 = qu.superket(np.array([[1,0],[0,0]],dtype = np.complex_))

def flatten(t):
    return [item for sublist in t for item in sublist.flatten()]

def get_C_coeff(circ,gate_set):
    """
    compute the coefficients (the row of C matrix) and ideal probability for a circuit;

    'circ' is a circuit, which is represented by a list of numbers;

    'gate_set'  is the gate set that we want to characterize (input as channel matrix).

    """
    coef=[]
    gates_list= []
    n_gates=len(gate_set)

    if not circ: # check null circuit
        coef.append(np.array([0 for i in range(12*n_gates)]))
        tot_u=np.eye(4)
    else:
        for gate in circ: # compose circuit
            gates_list.append(gate_set[gate])

        matrix_Gt_left=[]                       # 'matrix_Gt_left' is a list of such matrix G_{k:0}=G_k G_{k-1}...G_0, where k=0...L-1,
                                                # which denotes the product of gate maps from start to end.
        matrix_Gt_left.append(gates_list[0])
        for item in gates_list[1::]:
            matrix_Gt_left.append(np.dot(item,matrix_Gt_left[-1]))
        tot_u=matrix_Gt_left[-1]       # the corresponding unitary map of this circuit

        matrix_Gt_right=[]         # 'matrix_Gt_right' is a list of such matrix G_{L-1:k+1}= G_{L-1} ...G_{k+1}, where k=0...L-1,
                                   # which denotes the product of gate maps from end to start. But the last one (k=L-1) is the identity map.
        matrix_Gt_right.insert(0,np.eye(4))
        gates_list_inv=gates_list[::-1]
        for item in gates_list_inv[0:-1]:
            matrix_Gt_right.insert(0,np.dot(matrix_Gt_right[0],item))

        for n in range(n_gates):
            if n in circ:
                #print(n)
                pos_gate_n= np.where(np.equal(np.array(circ),n))[0] # find all the positions where the nth gate in the gate set is applied.
                #[print(qu.my_braket([matrix_Gt_left[item],z0_matrix,matrix_Gt_right[item]]).shape) for item in pos_gate_n]
                matrix_C=np.sum([qu.my_braket([matrix_Gt_left[item],z0_matrix,matrix_Gt_right[item]]) for item in pos_gate_n],axis=0)
                #matrix_C=np.sum([ matrix_Gt_left[item]@z0_matrix@matrix_Gt_right[item] for item in pos_gate_n],axis=0)
                                                      # 'z0_matrix' = |0>><<0|;  matrix_C= \sum_{k} G_{k:0} |0>><<0| G_{L-1:k+1}
                matrix_CT=matrix_C.transpose()
                coef.append(matrix_CT[1::])
            else:
                coef.append(np.array([0 for i in range(12)]))

    coef.append(np.array([1/np.sqrt(2)*tot_u[3,1],1/np.sqrt(2)*tot_u[3,2],1/np.sqrt(2)*tot_u[3,3],  # the coefficients for the errors of SPAM
                          1/np.sqrt(2),1/np.sqrt(2)*tot_u[1,3],1/np.sqrt(2)*tot_u[2,3],1/np.sqrt(2)*tot_u[3,3]]))
    ideal_p=qu.my_braket([z0,tot_u,z0])
    return np.array(list(flatten(coef))),ideal_p
