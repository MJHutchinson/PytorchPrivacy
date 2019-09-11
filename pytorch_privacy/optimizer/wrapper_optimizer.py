from abc import ABC, abstractmethod


class WrapperOptimizer(ABC):
    """
    Abstract class for wrapping up base PyTorch optimisers. This exists to allow either a standard or DP optimiser to
    be passed to a method or class with an identical interface, as extending the base optimisers to do differential
    privacy in PyTorch was not possible in a clean manner.
    """
    @abstractmethod
    def fit_batch(self, x, y):
        """
        Fit a single batch of x, y pairs using the wrapped optimiser, doing a single gradient step
        :param x:
        :param y:
        :return:
        """
        pass

    def get_logged_statistics(self):
        """
        Return any relevant statistics about the optimisation process that might be of interest for inspection. E.g.
        this could be statistics about the gradients, or the clipping of gradients for DP optimisers.
        :return: A nested dictionary structure of logged statistics
        """
        return {}

    def get_step_summary(self):
        """
        Return a simple summary of the step if this is desired. Designed to be able to report on the fly metrics for
        monitoring purposes
        :return: A nested dictionary of summary information
        """
        return {}