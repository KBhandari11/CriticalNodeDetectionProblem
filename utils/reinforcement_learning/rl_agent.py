import abc
import collections

StepOutput = collections.namedtuple("step_output", ["action", "probs"])


class AbstractAgent(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def __init__(self,
               session=None,
               observation_spec=None,
               name="agent",
               **agent_specific_kwargs):
    """Initializes agent.
    """

  @abc.abstractmethod
  def step(self, time_step, is_evaluation=False):
    """Returns action probabilities and chosen action at `time_step`.
    Returns:
      A `StepOutput` for the current `time_step`.
    """