"""Warm logp+gradient cost of individual state-space estimation features, on the BGP model.

One benchmark per feature, so each carries its own record: user-defined observation equations,
temporal aggregation (a quarterly series observed at annual frequency), and a mixed
log-linearisation that keeps the net rate in levels. Each builds and compiles its model in `setup`
(and warms it once), so the timed method measures only the fused logp+gradient evaluation — the
inner-loop cost the feature adds.
"""

from ._bgp import (
    frequency_change_model,
    mixed_loglin_model,
    observation_equations_model,
    warmed_logp_dlogp,
)


# Building the BGP logp+gradient compiles a large graph (tens of seconds); raise the per-benchmark
# timeout well above asv's 60s default so the untimed setup compile does not trip it.
TIMEOUT = 1200


class ObservationEquations:
    timeout = TIMEOUT

    def setup(self):
        self._f, self._q = warmed_logp_dlogp(observation_equations_model())

    def time_logp_dlogp(self):
        self._f(self._q)

    def peakmem_logp_dlogp(self):
        self._f(self._q)


class FrequencyChange:
    timeout = TIMEOUT

    def setup(self):
        self._f, self._q = warmed_logp_dlogp(frequency_change_model())

    def time_logp_dlogp(self):
        self._f(self._q)

    def peakmem_logp_dlogp(self):
        self._f(self._q)


class MixedLogLin:
    timeout = TIMEOUT

    def setup(self):
        self._f, self._q = warmed_logp_dlogp(mixed_loglin_model())

    def time_logp_dlogp(self):
        self._f(self._q)

    def peakmem_logp_dlogp(self):
        self._f(self._q)
