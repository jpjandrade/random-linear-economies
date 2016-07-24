import numpy as np
import cvxpy as cvx
import sys

EPS = 1e-3
eps = float(sys.argv[1])
M = int(sys.argv[2])


def gen_quenched(N):
    delta = 1.
    # Sample a random instance
    aux_vector = np.random.normal(scale=delta / M, size=(N, M))  #vetor r to generate quenched technologies
    xi = (aux_vector.T - (eps + np.sum(aux_vector, axis=1)) / M).T
    x_0 = np.random.exponential(size=M)
    return xi, x_0


def solve_s(xi, x_0, N, solver=None):
    s = cvx.Variable(N)
    obj = cvx.Maximize(cvx.sum_entries(cvx.log(x_0 + xi.T * s)))
    constr = [0 <= s]
    prob = cvx.Problem(obj, constr)
    prob.solve(solver=solver)
    return np.array(s.value).T[0]


def calc_utility(x):
    util = np.sum(np.log(x)) / len(x)
    return util


def calc_GDP(x,x_0):
    p = 1/x
    res = M*abs(x - x_0).dot(p)
    res /= 2*sum(p)
    return res


def calc_spread_reduction(s, xi, x_0):
    x0_avg = np.average(x_0)
    x0_spread = x_0 - x0_avg
    spread_reduction = np.sum(xi.dot(x0_spread)[s > EPS])
    return spread_reduction

n_list = np.arange(0.2, 1., 0.1)
# n_list = np.append(n_list, arange(2., 4., 0.5))
# n_list = np.append(n_list, arange(2., 5., 0.5))
# n_list = np.append(n_list, arange(5., 10., 1.))
# n_list = np.append(n_list, arange(10., 20., 1.))
n_len = len(n_list)
n_replicas = 10
s_pos_list = np.zeros((n_len, n_replicas))
s_avg_list = np.zeros((n_len, n_replicas))
x_avg_list = np.zeros((n_len, n_replicas))
gdp_avg_list = np.zeros((n_len, n_replicas))
utility_avg_list = np.zeros((n_len, n_replicas))
spread_reduction_avg_list = np.zeros((n_len, n_replicas))


for i, n in enumerate(n_list):

    n = n_list[i]
    N = int(n * M)
    print('n = ', n)

    for r in range(n_replicas):
        xi, x_0 = gen_quenched(N)
        try:
            s = solve_s(xi, x_0, N)
        except cvx.SolverError:
            s = solve_s(xi, x_0, N, solver='SCS')
        s = np.array(s).T
        x = x_0 + xi.T.dot(s)
        s_pos = np.sum(s > EPS) / N
        s_pos_list[i, r] = s_pos
        s_avg_list[i, r] = np.average(s)
        x_avg_list[i, r] = np.average(x)
        gdp_avg_list[i, r] = calc_GDP(x, x_0)
        utility_avg_list[i, r] = calc_utility(x)
        spread_reduction_avg_list[i, r] = calc_spread_reduction(s, xi, x_0)


np.savez('data_eps' + str(eps) + '.npz',
         n=n_list,
         s_pos=s_pos_list,
         s_avg=s_avg_list,
         x_avg=x_avg_list,
         gdp_avg=gdp_avg_list,
         utility_avg=utility_avg_list,
         spread_reduction=spread_reduction_avg_list
         )
