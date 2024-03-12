This is a reference to the Initial cox model:

- https://www.jstor.org/stable/2985181?read-now=1&seq=1#page_scan_tab_contents

The Cox Proportional-Hazards (PH) model is a statistical approach used in survival analysis, particularly when examining time-to-event data. In this continuous-time framework, the model consists of essential components. The baseline hazard function (λ0(t)) represents the risk of the event occurring at any given continuous time 't' for the reference group. Unlike parametric models, this function is flexible and adapts to the data. The inclusion of covariates (X) introduces factors influencing the hazard rate. Each covariate, denoted as 'X,' is associated with a coefficient ('β') that signifies the direction and magnitude of its impact on the hazard. The resultant hazard function (λ(t)) integrates the baseline hazard with the exponential function of the linear combination of covariate values and their coefficients. Mathematically, it is expressed as:

#### formula

An essential assumption of the Cox model is the proportional hazards assumption, suggesting that the hazard ratios for different covariate values remain constant over time.

To enhance efficiency, the model employs a partial likelihood that maximizes the likelihood of observed events, considering their chronological order. This is particularly useful when dealing with censored data, where events may not have occurred by a specific continuous time point.

The Cox model is adept at handling censoring, adjusting the likelihood function to account for instances where the event has not occurred by a particular continuous time. This flexibility makes it a robust tool for survival analysis, providing valuable insights into the dynamic interplay of covariates on the hazard function over continuous time.

The paper undergoes multiple levels of assumptions about the level of censoring and nature of the data the model is utilized on.

For the Cox PH model, the likelihood function considers the observed events and their order while accounting for censored data. The conditional likelihood is "conditional" because it focuses only on the observed events, given the observed and censored data up to a certain point in time.

Mathematically, the conditional likelihood function is expressed as the product of the conditional probabilities of events occurring at the observed failure times. It is maximized to find the parameter estimates (coefficients) that make the observed data most probable under the Cox model.
