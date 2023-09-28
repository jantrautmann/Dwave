import numpy as np
import dimod
from qiskit_nature.problems.sampling.protein_folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from qiskit_nature.problems.sampling.protein_folding.peptide.peptide import Peptide
from qiskit_nature.problems.sampling.protein_folding.protein_folding_problem import ProteinFoldingProblem
from qiskit_nature.problems.sampling.protein_folding.penalty_parameters import PenaltyParameters

def generate_hamiltonian(sequence):
    penalty_terms = PenaltyParameters(0.5,0.3, 0.2)

    side_chain_residue_sequences = [""]*len(sequence)   # Side chains 
    peptide = Peptide(sequence, side_chain_residue_sequences)

    mj_interaction = MiyazawaJerniganInteraction()

    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    qubit_op = protein_folding_problem.qubit_op() # The Hamiltonian

    return qubit_op

def solve_classically(hamiltonian):
    Hp=hamiltonian.to_matrix(massive=True)
    K = np.real(np.diag(Hp).tolist()).tolist()
    gs = bin(K.index(np.min(K)))

    return [np.int32(i) for i in gs[2:]]

def get_polynom(hamiltonian):
    poly = {}

    for (mask, val) in hamiltonian.primitive.to_list():
        # Count interacting terms
        int = mask.count("Z")
        if int == 1:
            ind = [i for i, x in enumerate(mask) if x == "Z"]
            poly[(ind[0],)] = np.real(val)
            # poly[(ind[0],)] = 1.0
        elif int == 2:
            ind = [i for i, x in enumerate(mask) if x == "Z"]
            poly[(ind[0], ind[1])] = np.real(val)
        elif int == 3:
            ind = [i for i, x in enumerate(mask) if x == "Z"]
            poly[(ind[0], ind[1], ind[2])] = np.real(val)
        elif int == 4:
            ind = [i for i, x in enumerate(mask) if x == "Z"]
            poly[(ind[0], ind[1], ind[2], ind[3])] = np.real(val)
        elif int == 5:
            ind = [i for i, x in enumerate(mask) if x == "Z"]
            poly[(ind[0], ind[1], ind[2], ind[3], ind[4])] = np.real(val)

    return poly


if __name__ == '__main__':
    # seq = "KLVFFA" # # 6 qubits
    # seq = 'APRLRFY' # 9 qubits
    seq = 'CYIQNCPLG' # 17 qubits
    hamiltonian = generate_hamiltonian(seq)

    gs = solve_classically(hamiltonian)
    n_qubits = len(gs)
    print('classical ground state : ', gs)
    poly = get_polynom(hamiltonian)
    bqm = dimod.make_quadratic(poly, 50.0, dimod.BINARY)

    print('number variables : ', bqm.num_variables)
    print('number interaction : ', bqm.num_interactions)
    print(bqm.quadratic)
    
    # sample = dimod.ExactSolver().sample(bqm).lowest()
    # # number of mapped qubits
    # mapped_qubits = len(sample.variables)
    # print('actual qubits required : ', mapped_qubits)

    # df =sample.to_pandas_dataframe(sample_column=True)
    # for i in range(len(sample)):
    #     state_dict = df['sample'].values[i]
    #     occupation = df['num_occurrences'].values[i]
    #     state = [state_dict[x] for x in range(n_qubits)]
    #     if state == gs:
    #         print('found ground state : ', state)
    #         print('with occupation : ', occupation)
    
