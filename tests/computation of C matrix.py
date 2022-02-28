def linear_model_coef(circ,gst=[np.eye(4),tg,hg]):
    """ 
    compute the coefficients (the row of C matrix) and ideal probability for a circuit;

    'circ' is a circuit, which is represented by a list of numbers;

    'gst'  is the gate set that we want to characterize.
    
    """
    coef=[]
    gates_list= []
    n_gates=len(gst)

    if circ==[]:
        coef.append(np.array([0 for i in range(12*n_gates)]))
        tot_u=np.eye(4)
    else:
        for gate in circ:
            gates_list.append(gst[gate])
        
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
                pos_gate_n= np.where(np.array(circ)==n)[0] # find all the positions where the nth gate in the gate set is applied.
                matrix_C=np.sum([my_prod_mat([matrix_Gt_left[item],z0_matrix,matrix_Gt_right[item]]) for item in pos_gate_n],axis=0)
                                                      # 'z0_matrix' = |0>><<0|;  matrix_C= \sum_{k} G_{k:0} |0>><<0| G_{L-1:k+1}
                matrix_CT=matrix_C.transpose()
                coef.append(matrix_CT[1::])
            else:
                coef.append([0 for i in range(12)])            
    
    coef.append([1/np.sqrt(2)*tot_u[3,1],1/np.sqrt(2)*tot_u[3,2],1/np.sqrt(2)*tot_u[3,3],  # the coefficients for the errors of SPAM
                  1/np.sqrt(2),1/np.sqrt(2)*tot_u[1,3],1/np.sqrt(2)*tot_u[2,3],1/np.sqrt(2)*tot_u[3,3]]) 
    ideal_p=my_prod_mat([z0,tot_u,z0])
    return np.array(list(flatten(coef))),ideal_p



def linear_model_coef_2q(circ,gst=gst2q_xyi):

# compute the coefficients and ideal probabilities for a two-qubit circuit
# Here we model the noise of each gate as a two-qubit map, i.e., 16 by 16 matrix with 240 free parameters due to Trace preserving constraints.
    coef=[]
    gates_list= []
    n_gates=len(gst)

    if circ==[]:
        coef.append([0 for i in range(240*n_gates)])
        coef.append([0 for i in range(240*n_gates)])
        coef.append([0 for i in range(240*n_gates)])
        tot_u=np.eye(16)
    else:
        for gate in circ:
            gates_list.append(gst[gate])
        
        matrix_Gt_left=[]                # 'matrix_Gt_left': G_{k:0} 
        matrix_Gt_left.append(gates_list[0])
        for item in gates_list[1::]:
            matrix_Gt_left.append(np.dot(item,matrix_Gt_left[-1]))
        tot_u=matrix_Gt_left[-1]

        matrix_Gt_right=[]            # 'matrix_Gt_right': G_{L-1:k+1} 
        matrix_Gt_right.insert(0,np.eye(16))
        gates_list_inv=gates_list[::-1]
        for item in gates_list_inv[0:-1]:
            matrix_Gt_right.insert(0,np.dot(matrix_Gt_right[0],item))

        for i in range(3):       # 'i' is the index for measurement outcome
            coefi=[]
            for n in range(n_gates):
                if n in circ:
                    pos_gate_n= np.where(np.array(circ)==n)[0]
                    matrix_C=np.sum([my_prod_mat([matrix_Gt_left[item],rho_eff_list[i],matrix_Gt_right[item]]) for item in pos_gate_n],axis=0) 
                    matrix_CT=matrix_C.transpose()
                    coefi.append(matrix_CT[1::])
                else:
                    coefi.append([0 for item in range(240)])            
            coef.append(list(flatten(coefi)))
    
    ideal_p_list=[]

    for j in range(1,16):
        coef[0].append(my_prod_mat([meas_eff_list[0],tot_u,com_basis2[j]]))
    for j in range(16):
        coef[0].append(my_prod_mat([com_basis2[j],tot_u,rho0_2q]))
    for j in range(32):
        coef[0].append(0)
    ideal_p_list.append(my_prod_mat([meas_eff_list[0],tot_u,rho0_2q]))


    for j in range(1,16):
        coef[1].append(my_prod_mat([meas_eff_list[1],tot_u,com_basis2[j]]))
    for j in range(16):
        coef[1].append(0)
    for j in range(16):
        coef[1].append(my_prod_mat([com_basis2[j],tot_u,rho0_2q]))
    for j in range(16):
        coef[1].append(0)
    ideal_p_list.append(my_prod_mat([meas_eff_list[1],tot_u,rho0_2q]))



    for j in range(1,16):
        coef[2].append(my_prod_mat([meas_eff_list[2],tot_u,com_basis2[j]]))
    for j in range(32):
        coef[2].append(0)
    for j in range(16):
        coef[2].append(my_prod_mat([com_basis2[j],tot_u,rho0_2q]))
    ideal_p_list.append(my_prod_mat([meas_eff_list[2],tot_u,rho0_2q]))


    return coef,ideal_p_list