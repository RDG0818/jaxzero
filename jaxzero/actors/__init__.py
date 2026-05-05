# ReplayBufferActor has no JAX dependency, safe to import at module level
from jaxzero.actors.replay_buffer_actor import ReplayBufferActor

# LearnerActor and DataActor have JAX dependencies, only import on demand
def __getattr__(name):
    if name == "LearnerActor":
        from jaxzero.actors.learner_actor import LearnerActor
        return LearnerActor
    elif name == "DataActor":
        from jaxzero.actors.data_actor import DataActor
        return DataActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ReplayBufferActor", "LearnerActor", "DataActor"]
