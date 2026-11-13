"""Independent isotropic TV least squares with an explicit positivity split."""

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def difference_operators(shape):
    if len(shape) != 2 or any(int(v) != v or v < 1 for v in shape):
        raise ValueError("Expected a positive two-dimensional grid")
    ny, nx = map(int, shape)
    n = nx * ny
    indices = np.arange(n).reshape(ny, nx)

    def operator(start, stop):
        start, stop = start.ravel(), stop.ravel()
        return sparse.csr_matrix((np.r_[-np.ones(start.size), np.ones(start.size)],
                                  (np.r_[start, start], np.r_[start, stop])), shape=(n, n))

    return operator(indices[:, :-1], indices[:, 1:]), operator(indices[:-1, :], indices[1:, :])


class TVSolver:
    """Minimize .5||B*u-y||^2 + beta*TV(u), subject to u >= lower_bound.

    B and y are already dimensionless. Difference operators use unit grid spacing;
    the acoustic adapter explicitly incorporates physical grid scaling in beta.
    """

    def __init__(self, matrix, shape, lower_bound=1e-6, max_iter=15000,
                 atol=1e-11, rtol=1e-7, rho=None, adaptive_rho=True, adapt_after=0):
        self.B = sparse.csr_matrix(matrix, dtype=np.float64)
        self.dx, self.dy = difference_operators(shape)
        self.n = self.dx.shape[0]
        if self.B.shape[1] != self.n or self.B.shape[0] == 0 or not np.isfinite(self.B.data).all():
            raise ValueError("Invalid forward matrix shape or values")
        if not np.isfinite(lower_bound) or lower_bound <= 0 or max_iter < 1:
            raise ValueError("Invalid positivity bound or iteration count")
        if atol <= 0 or rtol <= 0:
            raise ValueError("Tolerances must be positive")
        if not isinstance(adapt_after, int) or adapt_after < 0:
            raise ValueError("Adaptation delay must be a nonnegative integer")
        self.lower_bound, self.max_iter = float(lower_bound), int(max_iter)
        self.atol, self.rtol = float(atol), float(rtol)
        self.adaptive_rho = bool(adaptive_rho)
        self.adapt_after = adapt_after
        self.Bt = self.B.T.tocsr()
        self.AtA = (self.Bt @ self.B).tocsc()
        self.KtK = (self.dx.T @ self.dx + self.dy.T @ self.dy + sparse.eye(self.n)).tocsc()
        self.rho = float(rho) if rho is not None else max(float(self.AtA.diagonal().mean()), 1e-6)
        if not np.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("Invalid ADMM penalty")
        self._factors = {}

    def _factor(self, rho):
        if rho not in self._factors:
            self._factors[rho] = splu(self.AtA + rho * self.KtK)
        return self._factors[rho]

    def objective(self, solution, y, beta):
        residual = self.B @ solution - y
        return float(.5 * np.dot(residual, residual) + beta * np.hypot(self.dx @ solution, self.dy @ solution).sum())

    def solve(self, y, beta, initial=None):
        started = time.perf_counter()
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (self.B.shape[0],) or not np.isfinite(y).all():
            raise ValueError("Invalid measurement vector")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("Invalid TV weight")
        if initial is None:
            constant_projection = np.asarray(self.B.sum(axis=1)).ravel()
            denominator = np.dot(constant_projection, constant_projection)
            level = np.dot(constant_projection, y) / max(denominator, 1e-30)
            u = np.full(self.n, max(level, self.lower_bound))
        else:
            u = np.asarray(initial, dtype=np.float64).copy()
            if u.shape != (self.n,) or not np.isfinite(u).all():
                raise ValueError("Invalid initial field")
        z = np.maximum(u, self.lower_bound)
        d_x, d_y = self.dx @ u, self.dy @ u
        q_x, q_y, q_z = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)
        rhs_data = self.Bt @ y
        rho, converged = self.rho, False
        history = []
        for iteration in range(1, self.max_iter + 1):
            rhs = rhs_data + rho * (self.dx.T @ (d_x - q_x) + self.dy.T @ (d_y - q_y) + z - q_z)
            u = self._factor(rho).solve(rhs)
            gx, gy = self.dx @ u, self.dy @ u
            old_x, old_y, old_z = d_x, d_y, z
            vx, vy = gx + q_x, gy + q_y
            magnitude = np.hypot(vx, vy)
            shrink = np.maximum(1. - (beta / rho) / np.maximum(magnitude, 1e-30), 0.)
            d_x, d_y = shrink * vx, shrink * vy
            z = np.maximum(u + q_z, self.lower_bound)
            rx, ry, rz = gx - d_x, gy - d_y, u - z
            q_x += rx
            q_y += ry
            q_z += rz
            primal = float(np.sqrt(np.dot(rx, rx) + np.dot(ry, ry) + np.dot(rz, rz)))
            dual = float(rho * np.linalg.norm(self.dx.T @ (d_x - old_x) + self.dy.T @ (d_y - old_y) + z - old_z))
            ku_norm = np.sqrt(np.dot(gx, gx) + np.dot(gy, gy) + np.dot(u, u))
            v_norm = np.sqrt(np.dot(d_x, d_x) + np.dot(d_y, d_y) + np.dot(z, z))
            primal_tol = float(np.sqrt(3 * self.n) * self.atol + self.rtol * max(ku_norm, v_norm))
            dual_tol = float(np.sqrt(self.n) * self.atol + self.rtol * rho * np.linalg.norm(self.dx.T @ q_x + self.dy.T @ q_y + q_z))
            if not np.isfinite(primal + dual):
                raise FloatingPointError("Nonfinite ADMM residual")
            converged = primal <= primal_tol and dual <= dual_tol
            if iteration == 1 or iteration % 50 == 0 or converged or iteration == self.max_iter:
                history.append({"iteration": iteration, "primal": primal, "dual": dual,
                                "primal_tolerance": primal_tol, "dual_tolerance": dual_tol, "rho": rho})
            if converged:
                break
            if self.adaptive_rho and iteration >= self.adapt_after and iteration % 50 == 0:
                # Balance normalized residuals; rescale duals to preserve multipliers.
                ratio_p, ratio_d = primal / primal_tol, dual / dual_tol
                new_rho = rho
                if ratio_p > 10 * ratio_d:
                    new_rho = min(rho * 2., 1e6)
                elif ratio_d > 10 * ratio_p:
                    new_rho = max(rho / 2., 1e-12)
                if new_rho != rho:
                    q_x *= rho / new_rho
                    q_y *= rho / new_rho
                    q_z *= rho / new_rho
                    rho = new_rho
        return {"solution": z, "converged": bool(converged), "iterations": iteration,
                "objective": self.objective(z, y, beta), "primal_residual": primal,
                "dual_residual": dual, "primal_tolerance": primal_tol,
                "dual_tolerance": dual_tol, "rho": rho, "history": history,
                "elapsed_s": time.perf_counter() - started}
