Survival Analysis
Survival analysis studies the time until an event occurs in a system or environment. It addresses this by examining data distribution components:

Survival Function: Indicates probable survival past a given time.
Hazard Function: Shows the event rate at a specific time, given survival up to that point.
Cumulative Hazard: The accumulation of hazard over the lifetime of the data.
Likelihood Function: The combined probability of observations, given the data distribution's parameters.
Kaplan-Meier Estimator: Derives survival fraction over time from nonparametric attributes.
Outliers, which could skew results, are categorized as censored data:

Right Censoring: Event survival without observation.
Left Censoring: The event occurred before observation.
Interval Censoring: The event happened between two observations.
Truncation: Exclusion of a population group from the study because the event occurred before observations began.
Likelihood: Applied per datum (censored class) to calculate conditional probabilities of parameters.
This leads to discussing parametric, semi-parametric, and non-parametric components, detailing data distribution attributes over time:

Parametric Attributes: Defined by finite distribution parameters, e.g., mean, median.
Non-Parametric Attributes: Independent of distribution parameters, utilizing an infinite space, shown in methods like the sign-test.
Semi-Parametric: Analyzes both parametric and non-parametric attributes, focusing on nuisance parameters relative to finite ones.
Nuisance Parameter: Unspecified components of a function.
Covariates: Variables adjusted in models to isolate specific parameter effects.
Introducing the Cox Model or proportional hazards model, a semi-parametric approach:

Time-dependent covariates.
Baseline hazard level for covariates is non-parametric (nuisance).
Finite distribution parameters significantly influence hazard based on explainable covariates.
The hazard at time t varies from a baseline, remaining "proportional" across covariates. In high-dimensional settings, where covariates outnumber samples, lasso is used for variable selection and parameter regularization.
