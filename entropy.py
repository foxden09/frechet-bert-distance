def volume_of_unit_ball_log(d):
    d_over_2 = d / 2
    return d_over_2 * np.log(pi) - (loggamma(d_over_2 + 1))

def cross_entropy(N, M, k, nu_k, d):
    psi_k = digamma(k)
    c_bar = volume_of_unit_ball_log(d)
    inner_term = np.log(M) - psi_k + c_bar + d * np.log(nu_k)
    return (1 / N) * np.sum(inner_term)

def entropy(N, k, rho_k, d):
    psi_k = digamma(k)
    c_bar = volume_of_unit_ball_log(d)
    inner_term = np.log(N-1) - psi_k + c_bar + d * np.log(rho_k)
    return (1 / N) * np.sum(inner_term)
