import jax
import jax.numpy as jnp
import numpy as np
from pychastic.sde_problem import SDEProblem
from pychastic.sde_solver import SDESolver
import westpa


class SDEPropagator(westpa.Propagator):

    def __init__(self, *, problem, solver, seed=None):
        self.problem = problem
        self.solver = solver
        super().__init__(seed=seed)

    def propagate_block(self, segments):
        rng =  np.random.default_rng([segments[0].seg_id, segments[0].n_iter, self.seed])
        # see https://numpy.org/doc/2.2/reference/random/parallel.html#sequence-of-integer-seeds

        self.problem.x0 = np.array([segment.initial_state.coord for segment in segments])
        result = self.solver.solve_many(
            self.problem,
            n_trajectories=None,  # inferred from size of 'x0'
            progress_bar=False,
            seed=rng.integers(0, np.iinfo(int).max),
        )

        for i, segment in enumerate(segments):
            segment.final_state = westpa.State(coord=result['solution_values'][i, -1])

        return segments


class BrownianPropagator(SDEPropagator):

    def __init__(
        self,
        *,
        potential,
        inverse_temperature,
        friction_coefficient,
        step_size,
        steps,
        seed=None,
    ):
        super().__init__(
            problem=SDEProblem(
                a=lambda x: -jax.grad(potential)(jnp.atleast_1d(x)) / friction_coefficient,
                b=lambda x: np.atleast_2d(np.sqrt(2 / (inverse_temperature * friction_coefficient))),
                x0=np.array([np.nan]),
                tmax=steps * step_size,
            ),
            solver=SDESolver(dt=step_size),
            seed=seed,
        )
