===============
Master's Thesis
===============

This is my thesis for the Master of Science in Economics,
presented at the University of Bonn in the Winter Semester 2020/2021.

*Title*: On the Empirical Identification of Time Preferences in Discrete Choice
Dynamic Programming Models

*Supervisor*: Prof. Philipp Eisenhauer

*Author*: Sofia Badini

<hr />

--------
Abstract
--------

This thesis assesses the practical identification of the time preference parameters
in a discrete choice dynamic model of occupational choice, after introducing
empirically-motivated exclusion restrictions that influence the size of the agents'
choice set.

In particular, agents may or may not face future employment restrictions
that depend deterministically on their educational choices: Future-oriented agents
take the restrictions into account when deciding on their level of education.

The same identification strategy is used in a setting with exponential discounters
and in a setting where agents discounts quasi-hyperbolically and are completely naïve.

Time preference parameters in structural models of dynamic discrete choices are
underidentified, which is especially problematic for counterfactual analysis,
since information on time preferences are needed to make any statement about the
behavioral response of the agents to a policy intervention.

In the literature, both theoretical arguments for identification and empirical
identification strategies exploit, with mixed results, variables that leave the
per-period utility function unaffected, while being relevant to the agents'
decisions. The intuition is that comparing the behavioral response of similar agents
to different (expected) futures may reveal information on their time preferences.

<hr />

-----------
Replication
-----------

This project uses the template by Hans-Martin von Gaudecker (see
`here <https://econ-project-templates.readthedocs.io/en/stable/>`_ for more information),
therefore it is fully and automatically reproducible conditional on having
`conda <https://docs.conda.io/en/latest/>`_ installed.

<hr />

To reproduce, first navigate to the root of the project (the `master_thesis` folder).
Then, open your terminal emulator and run, line by line:

.. code-block:: zsh

    $ conda env create -f environment.yml
    $ conda activate master_thesis
    $ python waf.py configure
    $ python waf.py build
    $ python waf.py install


**Note: I haven't looked at this code since 2020, attempt to reproduce at your own risk.**
