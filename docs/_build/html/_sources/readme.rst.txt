PytorchPrivacy
==============

|Build Status| |Docs| |Coverage Status|

This library provides the functionality to easily perform Differentially
Private (DP) algorithms in python, with specific extensions to allow a
user to quickly apply DP to Pytorch gradient based optimisers.

Code organisation
^^^^^^^^^^^^^^^^^

The code is organised into 3 distinct sections.:

1. `dp\_query <https://github.com/MJHutchinson/PytorchPrivacy/tree/master/pytorch_privacy/dp_query>`__
   contains the code that defines generic DP queries (also called
   mechanisms), how they should process data to keep it private, and a
   ledger to keep a record of how data has been accessed in a DP way.

2. `analysis <https://github.com/MJHutchinson/PytorchPrivacy/tree/master/pytorch_privacy/analysis>`__
   contains the code to compute the resulting privacy bound from a
   series of DP queries stored in a ledger defined in
   `dp\_query <https://github.com/MJHutchinson/PytorchPrivacy/tree/master/pytorch_privacy/dp_query>`__.

3. `optimizer <https://github.com/MJHutchinson/PytorchPrivacy/tree/master/pytorch_privacy/optimizer>`__
   provides a way to wrap up Pytorch Optimiser with a model, a loss
   function, and a DP query to do differntially private optimisation of
   Pytorch models.

Additional utilities are found in
`utlis <https://github.com/MJHutchinson/PytorchPrivacy/tree/master/pytorch_privacy/utils>`__

A quick example
^^^^^^^^^^^^^^^

Further tutorials
^^^^^^^^^^^^^^^^^

References
^^^^^^^^^^

Some inspiration was drawn from the `Tensoflow Privacy
library <https://github.com/tensorflow/privacy>`__.

.. |Build Status| image:: https://travis-ci.org/MJHutchinson/PytorchPrivacy.svg?branch=master
   :target: https://travis-ci.org/MJHutchinson/PytorchPrivacy
.. |Docs| image:: https://readthedocs.org/projects/docs/badge/?version=latest
   :target: https://MJHutchinson.github.io/PytorchPrivacy
.. |Coverage Status| image:: https://coveralls.io/repos/github/MJHutchinson/PytorchPrivacy/badge.svg?branch=master
   :target: https://coveralls.io/github/MJHutchinson/PytorchPrivacy?branch=master
