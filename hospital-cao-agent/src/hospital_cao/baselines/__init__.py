from .policies import RandomPolicy, FIFOPolicy, AcuityFirstPolicy, EarliestDeadlinePolicy

ALL_BASELINES = {
    "random": RandomPolicy,
    "fifo": FIFOPolicy,
    "acuity_first": AcuityFirstPolicy,
    "earliest_deadline": EarliestDeadlinePolicy,
}

# ---------------------------------------------------------------------------
# Generative AI disclosure: Claude (Anthropic) was used to help draft the code in this project, as well as debugging syntax issues if the code broke.# 
# ---------------------------------------------------------------------------
