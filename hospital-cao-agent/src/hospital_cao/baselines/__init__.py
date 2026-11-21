from .policies import RandomPolicy, FIFOPolicy, AcuityFirstPolicy, EarliestDeadlinePolicy

ALL_BASELINES = {
    "random": RandomPolicy,
    "fifo": FIFOPolicy,
    "acuity_first": AcuityFirstPolicy,
    "earliest_deadline": EarliestDeadlinePolicy,
}
