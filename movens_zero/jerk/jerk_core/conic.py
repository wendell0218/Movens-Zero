

import numpy as np
from scipy import sparse
from scipy.signal import find_peaks
import clarabel

from .kinematics import FK_DOUBLE, physical, torch

if clarabel.__version__ != '0.11.1':
    raise RuntimeError('This optimizer requires clarabel==0.11.1')

CONFIGURATION = {
    'fps': 30.0,
    'outer_IRLS_SCP_iterations': 32,
    'jerk_epsilon_m_s3': 1.0,
    'variable_scale_m': 0.005,
    'root_fidelity_weight_in_jerk_objective': 0.3,
    'maximum_RAW_root_change_m': 0.03,
    'minimum_speed_scale_m_frame': 0.001,
    'minimum_order_margin_fraction': 0.02,
    'conflicting_minimum_distance_frames': 8,
    'normalized_order_L1_slack_penalty': 100.0,
    'trust_step_coordinate_bound_m': 0.004,
    'merit_backtracking_steps': 8,
    'stored_incumbent_check_every': 4,
    'seed_backtracking_steps': 12,
    'conic_solver': 'clarabel0.11.1',
    'conic_iterations': 100,
    'conic_tolerance': 1e-08,
    'conic_threads': 1,
    'slack_quadratic_regularizer': 1e-09,
    'pose_optimization': False,
    'exact_entire_initial_seed_events': True,
}


def optimize_segment(raw, seed):

    length = len(seed)
    seed_jerk, _, expected = physical(seed)
    if length < 4 or np.all(seed == seed[:1]):
        return seed.copy(), {'seed_Jerk': seed_jerk, 'Jerk': seed_jerk, 'short_or_static': True,
                             'exact_seed_events': True, 'seed_events': expected.tolist(), 'history': []}
    with torch.inference_mode():
        body = FK_DOUBLE.forward_body_positions(torch.from_numpy(seed.astype(np.float64))).numpy()
    d3 = sparse.diags([-np.ones(length-3), 3*np.ones(length-3), -3*np.ones(length-3), np.ones(length-3)],
                      [0, 1, 2, 3], shape=(length-3, length), format='csc')
    free_d3 = d3[:, 1:]
    residual = (d3 @ body.reshape(length, -1)).reshape(length-3, 30, 3)
    velocity = np.diff(body, axis=0)
    speed0 = np.linalg.norm(velocity, axis=-1).mean(axis=-1)
    speed_scale = max(float(np.median(speed0)), CONFIGURATION['minimum_speed_scale_m_frame'])
    scale, fidelity = CONFIGURATION['variable_scale_m'], CONFIGURATION['root_fidelity_weight_in_jerk_objective']
    minima = find_peaks(-speed0)[0]
    first, second = list(range(length-2)), list(range(1, length-1))
    for index, a in enumerate(minima):
        for b in minima[index+1:]:
            if b-a < CONFIGURATION['conflicting_minimum_distance_frames']:
                first.append(int(a))
                second.append(int(b))
    first, second = np.asarray(first), np.asarray(second)
    direction = np.sign(speed0[second]-speed0[first])
    margin = CONFIGURATION['minimum_order_margin_fraction']*np.abs(speed0[second]-speed0[first])/speed_scale
    rows = np.repeat(np.arange(length-1), 3)
    columns = np.arange(3*(length-1))
    count_x, count_s = len(columns), len(first)
    eye_x, eye_s = sparse.eye(count_x, format='csc'), sparse.eye(count_s, format='csc')
    zero_xs = sparse.csc_matrix((count_x, count_s))
    seed_root, raw_root = seed[:, :3].astype(float), raw[:, :3].astype(float)
    radius = max(0., .03-4*np.sqrt(3)*float(np.abs(np.spacing(raw[:, :3])).max()))
    soc_rows = np.repeat(4*np.arange(length-1), 3)+np.tile([1, 2, 3], length-1)
    root_soc = sparse.coo_matrix((-np.ones(count_x), (soc_rows, columns)), shape=(4*(length-1), count_x)).tocsc()
    root_b = np.column_stack((np.full(length-1, radius/scale), (seed_root[1:]-raw_root[1:])/scale)).reshape(-1)
    root_a = sparse.hstack((root_soc, sparse.csc_matrix((4*(length-1), count_s))), format='csc')
    settings = clarabel.DefaultSettings()
    settings.verbose, settings.max_threads = False, 1
    settings.max_iter = CONFIGURATION['conic_iterations']
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = CONFIGURATION['conic_tolerance']
    cones = [clarabel.NonnegativeConeT(2*count_s+2*count_x)]+[clarabel.SecondOrderConeT(4) for _ in range(length-1)]
    eps, jerk_scale = CONFIGURATION['jerk_epsilon_m_s3']/27000., 27000./(100*(length-3))

    def unpack(value):
        return np.vstack((np.zeros((1, 3)), value.reshape(length-1, 3)*scale))

    def order(value):
        vectors = velocity+np.diff(unpack(value), axis=0)[:, None, :]
        norms = np.linalg.norm(vectors, axis=-1)
        speed = norms.mean(axis=1)
        gradient = (vectors/np.maximum(norms[..., None], 1e-15)).mean(axis=1)
        jacobian = sparse.coo_matrix((np.r_[gradient.reshape(-1), -gradient[1:].reshape(-1)],
                       (np.r_[rows, rows[3:]], np.r_[columns, columns[3:]-3])),
                       shape=(length-1, count_x)).tocsr()*scale/speed_scale
        return (direction*(speed[second]-speed[first])/speed_scale-margin,
                sparse.diags(direction) @ (jacobian[second]-jacobian[first]))

    def merit(value):
        vectors = residual+(free_d3 @ (value.reshape(length-1, 3)*scale))[:, None, :]
        smooth_jerk = np.sqrt(np.square(vectors).sum(axis=-1)+eps**2).mean()*27000./100.
        regularizer = fidelity*float(np.square(value).sum())/(100*length)
        violation = CONFIGURATION['normalized_order_L1_slack_penalty']*np.maximum(-order(value)[0], 0).mean()
        return float(smooth_jerk+regularizer+violation)

    value, chosen, best, history = np.zeros(count_x), seed.copy(), seed_jerk, []
    for iteration in range(CONFIGURATION['outer_IRLS_SCP_iterations']):
        vectors = residual+(free_d3 @ (value.reshape(length-1, 3)*scale))[:, None, :]
        weights = 1/np.sqrt(np.square(vectors).sum(axis=-1)+eps**2)
        hessian = (jerk_scale*scale**2*free_d3.T @ sparse.diags(weights.mean(axis=1)) @ free_d3
                   +2*fidelity/(100*length)*sparse.eye(length-1, format='csc'))
        linear = (jerk_scale*scale*free_d3.T @ (weights[..., None]*residual).mean(axis=1)).reshape(-1)
        p = sparse.triu(sparse.block_diag((sparse.kron(hessian, sparse.eye(3)),
                               CONFIGURATION['slack_quadratic_regularizer']*eye_s)), format='csc')
        q = np.r_[linear, np.full(count_s, CONFIGURATION['normalized_order_L1_slack_penalty']/count_s)]
        f, jac = order(value)
        linear_a = sparse.vstack((sparse.hstack((-jac, -eye_s)),
                                  sparse.hstack((sparse.csc_matrix((count_s, count_x)), -eye_s)),
                                  sparse.hstack((eye_x, zero_xs)), sparse.hstack((-eye_x, zero_xs))), format='csc')
        trust = CONFIGURATION['trust_step_coordinate_bound_m']/scale
        linear_b = np.r_[f-jac @ value, np.zeros(count_s), value+trust, -value+trust]
        answer = clarabel.DefaultSolver(p, q, sparse.vstack((linear_a, root_a), format='csc'),
                                        np.r_[linear_b, root_b], cones, settings).solve()
        if str(answer.status) not in ('Solved', 'AlmostSolved'):
            history.append({'iteration': iteration, 'solver_status': str(answer.status), 'no_solution_retained': True})
            break
        proposal = np.asarray(answer.x[:count_x])
        previous_merit = merit(value)
        step, new_merit = 0., previous_merit
        for backtrack in range(CONFIGURATION['merit_backtracking_steps']+1):
            candidate_step = 2.**(-backtrack)
            candidate_merit = merit(value+candidate_step*(proposal-value))
            if candidate_merit < new_merit-1e-12:
                step, new_merit = candidate_step, candidate_merit
                break
        value = value+step*(proposal-value)
        trials = []
        if (iteration+1) % CONFIGURATION['stored_incumbent_check_every'] == 0 or step == 0 or iteration+1 == CONFIGURATION['outer_IRLS_SCP_iterations']:
            delta = unpack(value)
            for backtrack in range(CONFIGURATION['seed_backtracking_steps']+1):
                fraction = 2.**(-backtrack)
                candidate = seed.copy()
                candidate[:, :3] = (seed_root+fraction*delta).astype(raw.dtype)
                candidate[0] = raw[0]
                root_max = float(np.linalg.norm(candidate[:, :3].astype(float)-raw_root, axis=-1).max(initial=0))
                jerk, _, events = physical(candidate)
                exact = bool(np.array_equal(events, expected))
                feasible = root_max <= .03 and exact and np.isfinite(candidate).all()
                trials.append({'fraction': fraction, 'Jerk': jerk, 'root_change_max_m': root_max,
                               'exact_seed_events': exact, 'feasible': bool(feasible)})
                if feasible and jerk < best:
                    chosen, best = candidate, jerk
        history.append({'iteration': iteration, 'solver_status': str(answer.status), 'solver_iterations': int(answer.iterations),
                        'merit_before': previous_merit, 'merit_after': new_merit, 'accepted_step': step,
                        'continuous_order_max_violation': float(np.maximum(-order(value)[0], 0).max(initial=0)),
                        'incumbent_Jerk': best, 'stored_trials': trials})
        if step == 0:
            break
    return chosen, {'seed_Jerk': seed_jerk, 'Jerk': best, 'seed_events': expected.tolist(),
                    'selected_events': physical(chosen)[2].tolist(), 'exact_seed_events': True,
                    'history': history, 'root_SOCP_radius_m': radius, 'nonlinear_optimality_not_claimed': True}
