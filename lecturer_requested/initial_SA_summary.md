# Survival Analysis

Statistics that analysises the time unitl event ina a system (environment)

We try to address this by looking at certain components of the data distribution in this system:

- Survival Function: This aims to indicate the probable survival that is greater than the event time.
- Hazard function: denotes the event rate at time for conditional survival probability
- Cumulative Hazard: accumalation of hazard over life time of the distribution of data.
- Likelyhood Function: accumalated probability of observations as a function of parameter of the distribution of data.
- Kaplan-Meier: estimator function deriving the fraction of survival based on the nonparametric attributes overtime.

Because we have unexplainable outliers we don't want those data points affect our function calculations:

We categorize these outliers as censored data:

- Right Censoring: survival of event without observation
- Left Censoring: event occured prior to observation
- Interrval Censoring: event occurred between two observations
- Truncation: target group of population excluded from study. event occured prior to start of observations.
- LikelyHood: is relevant because it only applies if we calculate for each datum (censored class) to get the conditional probabability of parameters.

Which brings us to, Parameteric / semi-parametric / non-parametric components of our data distributions, which speaks
about the attributes of the data distribution overtime.

- Parametric Attributes: paramaterized by finite dimensional distribution families, an example being the mean and median etc.
- Non-Parametric Attributes: would then be the opposite of parameterized attributes by not being dependant on the distribution,
  or operating on an infinte distribution space, an example of non-parametric attributes is shown through the "sign-test"
- Semi-Parametric: both parametric and non parametric attributes where we analyse the nuisance parameter(non parametric)
  with respect to the finite dimensional parameters
- Nuisance Parameter: unspesified (left over) fraction of components of a function.
- Covariates: which operate under the above points are variables adjusted for in statistical models to isolate the effect of the parameters of interest.

Which brings us to the intro of the research at hand:

Cox Model ~ proportional hazards models

It is a semi-parametric model:

1. covariates are time-dependant
2. there is a baseline hazard levels of the covariates which is non-parametric (nuisance)
3. finite parameters of the distribution affect hazard in response to explainable covariates

a Subject's Hazard at time t varies from the baseline hazard

"proportianal" because the ration of hazards between covariates is constant.

with high-dimensionality, when number of covariates is relativly larger compared to the sample, the lasso method is
used to preforms variable selection and parameter regularisation
