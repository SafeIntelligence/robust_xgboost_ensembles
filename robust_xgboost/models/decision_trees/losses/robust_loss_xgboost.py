"""
Numba optimized function to compute the robust loss over a single threshold.
Relies on a linear relaxation of the robust splitting criterion of XGBoost. 

The linear relaxation is formulated as:

L = (A + p)^2 / (C + q) + (B + T - p)^2 / (D + Q - q)

where:
- A:    The sum of the first derivatives of the loss function of 
        unambiguous points on the left child node.
- B:    The sum of the first derivatives of the loss function of
        unambiguous points on the right child node.
- C:    The sum of the second derivatives of the loss function of
        unambiguous points on the left child node.
- D:    The sum of the second derivatives of the loss function of
        unambiguous points on the right child node.
- T:    The total sum of the first derivatives of the loss function
        of all ambiguous points.
- Q:    The total sum of the second derivatives of the loss function
        of all ambiguous points.
- p:    The sum of the first derivatives of ambiguous points going left.
        (variable to be optimized)
- q:    The sum of the second derivatives of ambiguous points going left
        (variable to be optimized)
"""


from numba import njit

TOL = 1e-9
TOL_ROB_LOSS = 1e-9

@njit(error_model='numpy', cache=True)
def f(p, q, A, B, C, D, T, Q):
    A_plus_p = A + p
    B_plus_T_minus_p = B + T - p
    C_plus_q = C + q 
    D_plus_Q_minus_q = D + Q - q
    return (A_plus_p * A_plus_p) / (C_plus_q + TOL) + (B_plus_T_minus_p * B_plus_T_minus_p) / (D_plus_Q_minus_q + TOL)

@njit(error_model='numpy', cache=True)
def check_feasibility_and_minimum_f(
    A, B, C, D, T, Q, p, q, p_min, p_max, G1, G2,
    current_minimum_f, current_minimum_p, current_minimum_q
):
    
    minimum = False
    
    if p < p_min - 1e-7:
        return False, current_minimum_f, current_minimum_p, current_minimum_q
        
    elif p > p_max + 1e-7:
        return False, current_minimum_f, current_minimum_p, current_minimum_q
    
    if q < 0 - 1e-7:
        return False, current_minimum_f, current_minimum_p, current_minimum_q
        
    elif q > Q + 1e-7:
        return False, current_minimum_f, current_minimum_p, current_minimum_q
    
    # if G1 > 0:
    
    if G1*q > p + 1e-7:
        return False, current_minimum_f, current_minimum_p, current_minimum_q
        
    elif G2*q < p - 1e-7:
        return False, current_minimum_f, current_minimum_p, current_minimum_q
        
    
        
    if p > T - G1*(Q-q) + 1e-7:
        return False, current_minimum_f, current_minimum_p, current_minimum_q
        
    elif p < T - G2*(Q-q) - 1e-7:
        return False, current_minimum_f, current_minimum_p, current_minimum_q
    
        
    f_val = f(p, q, A, B, C, D, T, Q)
    
    if f_val < current_minimum_f:
        minimum = True
        current_minimum_f = f_val
        current_minimum_p = p
        current_minimum_q = q
    
    return minimum, current_minimum_f, current_minimum_p, current_minimum_q
    
@njit(error_model='numpy', cache=True)
def compute_robust_loss(
    A,
    B,
    C,
    D,
    T,
    Q,
    p_min, 
    p_max, 
    min_gi_amb,
    max_gi_amb,
    min_hi_amb,
    max_hi_amb,
):
    
    optimal_p = 0
    optimal_q = 0
    optimal_f = 1e9  
    
    # precomputing some common terms for efficiency
    AD = A*D
    AQ = A*Q
    BC = B*C
    CT = C*T
    BQ = B*Q
    QT = Q*T
    CD = C*D
    CQ = C*Q
    DQ = D*Q
    DT = D*T
    
    C_plus_D_plus_Q = C + D + Q
    A_plus_B_plus_T = A + B + T  
    
    if min_gi_amb > 0:
        G1 = min_gi_amb/(max_hi_amb + 0* TOL_ROB_LOSS)
        
    else:
        G1 = min_gi_amb/(min_hi_amb + 0*TOL_ROB_LOSS)
        
        
    if max_gi_amb > 0:
        G2 = max_gi_amb/(min_hi_amb + 0*TOL_ROB_LOSS)
        
    else:
        G2 = max_gi_amb/(max_hi_amb + 0*TOL_ROB_LOSS)

    
    # check if the analytical solution is feasible
    
    
    # 1.1 p = G1*q
    
    q_star = (AD + AQ - BC - CT)/(A + B - G1*(C_plus_D_plus_Q) + T + TOL_ROB_LOSS)
    p_star = G1 * q_star
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 1.2 p = G2*q
    
    q_star = (AD + AQ - BC - CT)/(A + B - G2*(C_plus_D_plus_Q) + T + TOL_ROB_LOSS)
    p_star = G2 * q_star
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 1.3 p = T - G1*(Q-q)
    
    q_star = (AD + AQ - BC - G1*Q*(C_plus_D_plus_Q) + DT + QT) / (A + B - G1*(C_plus_D_plus_Q) + T + TOL_ROB_LOSS)
    p_star = T - G1*(Q-q_star)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 1.4 p = T - G2*(Q-q)
    
    q_star = (AD + AQ - BC - G2*Q*(C_plus_D_plus_Q) + DT + QT) / (A + B - G2*(C_plus_D_plus_Q) + T + TOL_ROB_LOSS)
    p_star = T - G2*(Q-q_star)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # check all of the boundaries, and see if they are feasible 
    
    # 2.1 p = p_min
    
    p_star = p_min
    # q_star = (AD + AQ - BC - CT + p_min*(C_plus_D_plus_Q)) / (A_plus_B_plus_T + TOL_ROB_LOSS)
    q_star = (AD + AQ + BC + CT + p_min*(D+Q-C)) / (A - B - T + 2*p_min + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 2.2 p = p_max
    
    p_star = p_max
    # q_star = (AD + AQ - BC - CT + p_max*(C_plus_D_plus_Q)) / (A_plus_B_plus_T + TOL_ROB_LOSS)
    q_star = (AD + AQ + BC + CT + p_max*(D+Q-C)) / (A - B - T + 2*p_max + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 2.3 q = 0
    
    q_star = 0
    p_star = (-AD - AQ + BC + CT) / (C_plus_D_plus_Q + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 2.4 q = Q
    
    q_star = Q
    p_star = (-AD + BC + BQ + CT + QT)/(C+D+Q + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 2.5 G1*q = p 
    
    q_star = (AD + AQ - BC - CT)/(A + B - G1*(C_plus_D_plus_Q) + T + TOL_ROB_LOSS)
    p_star = G1 * q_star
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    q_star = (AD + AQ + BC - 2*G1*(CD + CQ) + CT)/(A - B + G1*(-C + D + Q) - T + TOL_ROB_LOSS)
    p_star = G1 * q_star
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 2.6 G2*q = p
    
    q_star = (AD + AQ - BC - CT)/(A + B - G2*(C_plus_D_plus_Q) + T + TOL_ROB_LOSS)
    p_star = G2 * q_star
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    q_star = (AD + AQ + BC - 2*G2*(CD + CQ) + CT)/(A - B + G2*(-C + D + Q) - T + TOL_ROB_LOSS)
    p_star = G2 * q_star
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 2.7 p = T - G1*(Q-q)
    
    q_star = (AD + AQ + BC - G1*(2*CD + CQ + DQ + Q*Q) + DT + QT)/(A - B + G1*(-C + D + Q) + T + TOL_ROB_LOSS)
    p_star = T - G1*(Q-q_star)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    q_star = (AD + AQ - BC - G1*(CQ + DQ + Q*Q) + DT + QT)/(A + B - G1*(C_plus_D_plus_Q) + T + TOL_ROB_LOSS)
    p_star = T - G1*(Q-q_star)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 2.8 p = T - G2*(Q-q)
    
    q_star = (AD + AQ + BC - G2*(2*CD + CQ + DQ + Q*Q) + DT + QT)/(A - B + G2*(-C + D + Q) + T + TOL_ROB_LOSS)
    p_star = T - G2*(Q-q_star)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    q_star = (AD + AQ - BC - G2*(CQ + DQ + Q*Q) + DT + QT)/(A + B - G2*(C_plus_D_plus_Q) + T + TOL_ROB_LOSS)
    p_star = T - G2*(Q-q_star)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    
    # check all of the corners and see if they are feasible
    
    # 3.1 p = p_min, q = 0
    
    p_star = p_min
    q_star = 0
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.2 p = p_min, q = Q
    
    p_star = p_min
    q_star = Q
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.3 p = p_max, q = 0
    
    p_star = p_max
    q_star = 0
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.4 p = p_max, q = Q
    
    p_star = p_max
    q_star = Q
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.5 p = G1*q, p = G2*q + T - G2*Q
    
    q_star = (T - G2*Q)/(G1 - G2 + TOL_ROB_LOSS)
    p_star = G1*q_star
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.6 p = G1*q, p = G2*q
    
    q_star = 0.0
    p_star = 0.0
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.7 p = G2*q, p = G1*q + T - G1*Q
    
    q_star = (T - G1*Q)/(G2 - G1 + TOL_ROB_LOSS)
    p_star = G2*q_star
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.8 p = G1*q + T - G1*Q, p = G2*q + T - G2*Q
    
    q_star = Q
    p_star = T
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.9
    
    p_star = 0.0
    q_star = 0.0
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.10
    
    p_star = G1*Q
    q_star = Q
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.11
    
    p_star = p_min
    q_star = G1 / (p_min + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.12
    
    p_star = p_max
    q_star = G1 / (p_max + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.13
    
    p_star = 0.0
    q_star = 0.0
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.14
    
    p_star = G2*Q
    q_star = Q
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.15
    
    p_star = p_min
    q_star = G2 / (p_min + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.16 
    
    p_star = p_max
    q_star = G2 / (p_max + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.17
    
    p_star = T - G1*Q
    q_star = 0.0
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.18
    
    p_star = T
    q_star = Q
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.19
    
    p_star = p_min
    q_star = (p_min + G1*Q - T) / (G1 + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.20
    
    p_star = p_max
    q_star = (p_max + G1*Q - T) / (G1 + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.21
    
    p_star = T - G2*Q
    q_star = 0.0
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.22
    
    p_star = T
    q_star = Q
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.23
    
    p_star = p_min
    q_star = (p_min + G2*Q - T) / (G2 + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    # 3.24
    
    p_star = p_max
    q_star = (p_max + G2*Q - T) / (G2 + TOL_ROB_LOSS)
    
    _, optimal_f, optimal_p, optimal_q = check_feasibility_and_minimum_f(A, B, C, D, T, Q, p_star, q_star, p_min, p_max, G1, G2, optimal_f, optimal_p, optimal_q)
    
    return optimal_p, optimal_q, optimal_f

