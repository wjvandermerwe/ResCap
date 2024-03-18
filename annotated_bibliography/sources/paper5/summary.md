Objective: This paper addresses the challenge of analyzing survival outcomes in the presence of non-proportional hazards, a common issue when comparing censored survival times across treatment groups in observational health care studies. The standard Cox proportional hazards model, which assumes constant hazard ratios over time, often does not fit real-world data accurately.

Background: Non-proportional hazards violate the Cox model's assumption, leading researchers to explore alternative models. While non-parametric methods like the Kaplan-Meier estimator do not assume a specific survival curve shape, they typically do not incorporate covariates directly. This limitation motivates the use of parametric and semi-parametric regression approaches that can handle non-proportional hazards and adjust for covariates.

Methods Reviewed:

Non-Parametric Approaches: Including the Kaplan-Meier and Nelson-Aalen estimators, which accommodate non-proportionality without making assumptions about the survival curves' functional form.
Parametric and Semi-Parametric Models: Such as modified Cox models with time-dependent hazards, Accelerated Failure Time (AFT) models, and proportional odds models, each offering a way to account for non-proportional hazards. The paper also discusses the utility of pseudo-observations.
Propensity Score Weighting: The use of Inverse Probability of Treatment Weighting (IPTW) with these models to address confounding in observational studies.
Key Findings from Simulations and Clinical Studies:

The IPTW Kaplan-Meier method, a simple and computationally efficient approach, performed well across various scenarios, effectively addressing non-proportionality with minimal increase in standard errors.
Alternative propensity score weights, like variance stabilized weights, show promise, especially in scenarios where treatment distribution is uneven.
Flexible procedures for estimating propensity scores, potentially using ensemble machine learning methods, could offer improvements in handling complex covariate effects.
Practical Implications: The findings suggest that IPTW Kaplan-Meier curves are a viable method for estimating treatment effects in studies with non-proportional hazards, offering a balance between computational simplicity and robustness against incorrect inferences.

Conclusion: The paper contributes to survival analysis literature by comparing different methods' performance in non-proportional hazards contexts, highlighting the IPTW Kaplan-Meier method's practicality and efficiency. Future research directions include exploring more sophisticated propensity score estimation techniques and further validating these methods in diverse clinical scenarios.
