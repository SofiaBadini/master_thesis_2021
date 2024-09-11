"""Empirical moments to compute for exponential and hyperbolic model specification
respectively.

"""

from src.library.compute_moments import calc_very_restricted_choice_probabilities
from src.library.compute_moments import calc_restricted_choice_probabilities
from src.library.compute_moments import calc_unrestricted_choice_probabilities
from src.library.compute_moments import calc_very_restricted_wage_distribution
from src.library.compute_moments import calc_restricted_wage_distribution
from src.library.compute_moments import calc_unrestricted_wage_distribution

calc_moments = {
    "exp": {
        "Choice Probabilities Restricted": calc_restricted_choice_probabilities,
        "Choice Probabilities Unrestricted": calc_unrestricted_choice_probabilities,
        "Wage Distribution Restricted": calc_restricted_wage_distribution,
        "Wage Distribution Unrestricted": calc_unrestricted_wage_distribution,
    },
    "hyp": {
        "Choice Probabilities Very Restricted": calc_very_restricted_choice_probabilities,
        "Choice Probabilities Restricted": calc_restricted_choice_probabilities,
        "Choice Probabilities Unrestricted": calc_unrestricted_choice_probabilities,
        "Wage Distribution Very Restricted": calc_very_restricted_wage_distribution,
        "Wage Distribution Restricted": calc_restricted_wage_distribution,
        "Wage Distribution Unrestricted": calc_unrestricted_wage_distribution,
    },
}
