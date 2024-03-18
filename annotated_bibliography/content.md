% This is annote.bib
% Author: Ayman Ammoura
% A demo for CMPUT 603 Fall 2002.
% The order of the following entries is irrelevant. They will be sorted according to the
% bibliography style used.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
@article{cox_regression_1972,
title = {Regression Models and Life-Tables},
volume = {vol. 34},
url = {http://www.jstor.org/stable/2985181},
series = {Series B (Methodological)},
pages = {187--220},
issue = {no. 2},
journal= {Journal of the Royal Statistical Society},
author = {Cox, D. R.},
year = {1972},
annote = {\textbf{Aim:}
Foundational component of the research topic, it presents the intial formulation and proofs of the Cox model.
\\\\
\textbf{Style/Type:}
journal article, theoretical
\\\\
\textbf{Cross references:}
The Cox model, has been widely adopted and adapted as seen in the preceeding texts, and is a core models used frequently for survival analysis. We see its effectiveness to capture risk ratios and ability to accomodate censoring.
\\\\
\textbf{Summary:}
The paper introduces a statistical model, which acts as an extension to prior work formalised as the kaplan-meier estimator, by exploring time to event data (life tables). The major benefit, speaks to the concept of censored data, which is a known concept in survival analysis, there is missing information within the data, specifically, event occurrence without observation on a continuous time scale. The proposition consists of covariates, known as attributes regarding a unit in a distribution of data, which is associated with a coefficient Beta scaling the impact of said covariates, this product is then bound by the baseline hazard which gives the risk of event occurrence at a specific time.

\[h(t|X) = h_0(t) \exp(\beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_n X_n)\]

Hazard in other words being the estimated conditional probabilities, inline with the observed conditional frequencies of events. An essential assumption of the Cox model is the proportional hazards assumption, suggesting that the hazard ratios for different covariates remain constant over time we see this for two events observations,

\[\frac{h(t|X_1)}{h(t|X_2)} = \frac{h_0(t) \exp(\beta^T X_1)}{h_0(t) \exp(\beta^T X_2)} = \frac{\exp(\beta^T X_1)}{\exp(\beta^T X_2)} = \exp(\beta^T (X_1 - X_2))\]

The model efficiently deals with censoring, by adjusting the likelihood function for observations where event occurrence did not happen in a particular continuous time, and by maximising the likelihood of all observed events, it is possible to estimate the coefficients which can work the best under the cox formulation.

\[L(\beta) = \prod*{i: \delta_i = 1} \frac{\exp(\beta^T X_i)}{\sum*{j \in R(t_i)} \exp(\beta^T X_j)}\]

              }

}
%%

@article{tibshirani_regression_1996,
title = {Regression Shrinkage and Selection via the Lasso},
volume = {vol. 58},
url = {https://www.jstor.org/stable/2346178},
series = {Series B (Methodological)},
pages = {267--88},
issue = {no. 1},
journal = {Journal of the Royal Statistical Society.},
author = {Tibshirani, Robert},
year = {1996},
annote = {\textbf{Aim:}
Another foundational component of the research, it presents a method to make regression models more understandable and performant.
\\\\
\textbf{Style/Type:}
journal article, theoretical
\\\\
\textbf{Cross references:}
It is provided that this method could be used for improving the interpratebility of effects of the underlying parameters, it is appropriate in combination with a regression model like the cox model.
\\\\
\textbf{Summary:}
The paper presents proof for a new method which aims to solve two objective issues with Ordinary Least Squares (OLS) estimates: prediction accuracy and model interpretability, for scenarios where the number of predictors is large or when there is collinearity among predictors. The method minimises the residual sum of squares in relation to the sum of absolute values of the coefficients being less than a constant.

\[\hat{\beta}^{lasso} = \arg \min*{\beta} \left\{ \frac{1}{2N} \sum*{i=1}^{N} (y*i - \beta_0 - \sum*{j=1}^{p} X\_{ij}\beta_j)^2 \right\}\]

bound by

\[\sum\_{j=1}^{p} |\beta_j| \leq t.\]

The benefit of the model being that on a continuous scale the method has the ability to set some coefficients exactly to zero, which excludes feature contributions and thereby performs variable selection, which yields improved interpretability. In the evaluation section of the paper lasso is compared with subset selection, ridge regression and the non-negative garotte method, which shows competitive, prediction accuracy in various scenarios characterised by the number and magnitude of effect sizes of the predictors. Specifically, it performs well in settings with a moderate number of moderate-sized effects and when there's a large number of small effects, indicating its versatility across different regression contexts.
}
}
%%

@article{ishwaran_random_2008,
title = {Random Survival Forests},
volume = {2},
url = {DOI 10.1214/08-AOAS169},
pages = {841--860},
number = {3},
journal = {Annals of Applied Statistics},
author = {Ishwaran, Hemant and Kogalur, Udaya B and Blackstone, Eugene H and Lauer, Michael S.},
year = {2008},
annote = {\textbf{Aim:}
The final foundational component, providing a method for combining machine learning and statistical parts.
\\\\
\textbf{Style/Type:}
journal article, theoretical
\\\\
\textbf{Cross references:}
Operating under the same principle of producing hazards for the survival distrbutions, it would be appropriate to compare this model to a cox regression method.
\\\\
\textbf{Summary:}

Random survival forests is an extension of random forests, which has the ability to handle right-censored data and aim to estimate the appropriate survival function. Consisting of an ensemble of trees, which are grown from a bootstrap sample, and each node of underlying trees, consist of covariates. Per node splitting criteria is conditional to survival time and censoring, where by node “impurity” is determined by the survival differences. Methods like logrank, conservation of events splitting rule, and random log rank are used. Terminal nodes are the result of saturated splitting criteria, with each endpoint having d-dimensional covariates of the individuals contained. A key component of the model is the conservation of events principle, which is used to define a type of predicted outcome, namely ensemble mortality,

\[\hat{M}^{_}*{e,i}(t) = \sum*{j=1}^{n} H^{_}_{e}(t_{j}|X\_{i})\]

which is derived from the cumulative hazard function (CHF) using the Nelson-Aalen estimator. All terminal nodes share the estimated hazard function.

\[\hat{H}_{h}(t) = \sum_{t_i,h \leq t} \frac{d_i,h}{n_i,h}\]

Another key concept is the out of bag (OOB) which is a validation subset. The OOB error is calculated on the ensemble survival function with regards to the observed data using metrics like concordance.

\[H^{\*_}*{e}(t|X*{i}) = \frac{\sum*{b=1}^{B}\,I*{i,b}H^{_}_{b}(t|X_{i})}{\sum*{b=1}^{B}\,I*{i,b}}\]

Prediction error metrics, like concordance index which calculates the permissible pairs per node and OOB prediction error, are used for accuracy metrics. Variable Importance (VIMP) is assessed by looking at each predictor variable in the sample and assessing the impact on prediction error, increases in error indicating importance. The paper puts forward an approach to deal with missing data, outlining the shortcomings of prior methods like replacing missing values with distribution medians, and for categorical data replacing with most frequent occurrences. The method is called adaptive tree imputation, and relies on the OOB data set to determine missing data, for both continuous or integer values. A Key benefit of the model is within its ability to capture survival functions for an individual in the distribution by estimating its survival function across all trees where the individual is captured in terminal nodes. Another benefit is that the model is well suited for high dimensional data because of the random subset selection process, which helps mitigate overfitting. Due the permutative nature of the ensemble bound to the brevity of the underlying data distribution, the model is computationally demanding, and although the model can yield variable information, it might be difficult to interpret the final resulting model
}
}
%%

@article{haider*effective_2018,
title = {Effective Ways to Build and Evaluate Individual Survival Distributions},
volume = {21},
doi = {arXiv:1811.11347},
pages = {18--772},
number = {2020},
journal = {Journal of Machine Learning Research ({JMLR})},
author = {Haider, Humza and Hoehn, Bret and Davis, Sarah and Greiner, Russell},
year = {2018},
month ={11},
day = {29},
annote = {\textbf{Aim:}
Practical application of the research foundational models, methodology and adaptations of the foundational models of the research objective.
\\\\
\textbf{Style/Type:}
journal article, empirical
\\\\
\textbf{Cross references:}
The paper explores, spesific situations and typing of models, which produce outputs that can provide meaning applicable to different observers. It relates to the priors, due to the underlying methods used aligning with previous literature.
\\\\
\textbf{Summary:}
This paper is a detailed exploration of the differences and benefits of modelling individual survival distributions as compared to the current models that act on a general population. It is provided that standard methods fall short in assessing effects for an individual and suggest the use of Individual Survival Distribution (ISD) models. The key outline being that the paper presents five classes of tools, One Value individual risk models (\([R,1, V*{i}]\)), which is inline with the cox model which provides risk scores for each patient, single time group risk predictors (\([R,1_{t^*},g]\)), such as prognostic scales, assigning patients to risk groups, single time individual probabilistic predictors (\([R,1_{t^*},g]\)), like the Gail Model, which gives the probability at a specific time point for an individual, Group survival distributions (\([P,\infty,g]\)), like Kaplan-Meier curves, provides population level probabilities over time, and lastly individual survival distribution models(\([P,\infty,i]\)), which are a group of models outlined in the paper for evaluation, models like cox extensions are used which generates probability curves per patient across all time. For evaluation the authors highlights concordance, which compares predicted risk scores with outcomes, L1-loss which yields the average absolute difference between predicted and outcome survival times, 1-calibration which checks specific predicted time point probabilities are well calibrated to actual survival rates, Integrated Brier score (IBS), which indicates calibration and discrimination measures. The authors propose an evaluation method called D-calibration, a method that aims at indicating if probability estimates are meaningful across survival curves. The method operates over subsets (\(D\_{\Theta}\)) across the entire data distribution.

\[\frac{|D\_{\Theta}([a, b])|}{|D|} = b - a\]

The study provides empirical results after applying the above groups with relevant methods(Cox Kalbfleisch-Prentice, ATF, Random survival forests multi task regression) which show the the MLTR method outperforms most models across the different measures, and demonstrate the concept of D-calibration extensively proving that some of the methods demonstrate cross distribution calibration. The authors conclude that ISD can be very useful, and highlight impacts of some limitations of models and the successive investigation still needed.
}
}
%%

@misc{handorf_analysis_2021,
title = {Analysis of survival data with non-proportional hazards: A comparison of propensity score weighted methods},
url = {https://doi.org/10.48550/arXiv.2009.00785},
author = {Handorf, Elizabeth A. and Smaldone, Marc and Movva, Sujana and Mitra, Nandita},
date = {2021-02-02},
annote = {\textbf{Aim:}
Exploring common issues with survival data, provides implementation insight of applying adaptation of foundational methods and simulation data.
\\\\
\textbf{Style/Type:}
preprint, empirical
\\\\
\textbf{Cross references:}
The paper trys to bridge the gap for some of the underlying boundries the foundational methods propose, whilst providing methodology for statistical simulations that is close to real world examples.  
 \\\\
\textbf{Summary:}
The paper analysis scenarios where the main assumption of the cox model, namely proportional hazards, are violated. A broad look at numerous methods, all operating in the context of adjusting for differences between treatment groups using Inverse Probability of Treatment Weighting(IPTW) which is based on propensity score.

\[e = P(Z = 1 | X)\]
Propensity score $e$ represents the probability of receiving treatment ($Z=1$) given covariates $X$

The authors point out that the cox constant hazard ratios often fall short of real world data. Propensity score being the probabilistic outcome of receiving treatment in relation to the covariates of a unit. This helps with “causal inference” which adjusts the occurrence of confounding during observational studies. Confounding being described as the choice of treatment that is informed by factors that are also associated with survival. IPTW creates synthetic samples, which helps covariates stay unrelated to the treatment assignment. The authors use the Average treatment effect (ATE) calculation to determine the estimated propensity for a given subject i.
\[\hat{w}\_i = \frac{Z_i}{\hat{e}\_i} + \frac{1 - Z_i}{1 - \hat{e}\_i}\]

IPTW $\hat{w}_i$ adjusts individual $i$'s contribution based on their probability of receiving treatment $\hat{e}_i$, where $Z_i$ indicates treatment status.

\[ATE = E(Y_1 - Y_0)\]
ATE measures the average effect of treatment by comparing expected outcomes $Y_1$ (treated) and $Y_0$ (untreated).

The limitations described above motivates the use of non-parametric, like Kaplan-Meier(KM) estimators, which doesn’t make assumptions about survival outcomes.

\[\hat{S}(t) = \prod*{t_j \leq t} \left( 1 - \frac{\hat{d}^{w}*{j}}{\hat{n}^{w}\_{j}} \right)\]
Survival function $\hat{S}(t)$ estimates the probability of survival up to time $t$, adjusted for IPTW.

where,

\[\hat{d}_{w_j} = \sum_{i: T*i = t_j} \hat{w}\_i \delta_i\]
Weighted number of events $\hat{d}*{w_j}$ at time $t_j$, where $\delta_i$ indicates occurrence of the event for individual $i$.

Parametric and semi-parametric, like accelerated failure time (AFT) and its gamma extension model and cox with time-dependent hazards serving the purpose of accounting for non proportional hazards (relaxing proportionality). The authors also use a method called pseudo observations with propensity score weighting, which operates under the missing data premise to help with censoring and estimating survival.

\[\hat{P}(Y | Z, t) = \frac{1}{n*Z} \sum_i \hat{w}\_i \hat{\theta}*{Zi}\]

Probability of outcome $Y$ given treatment $Z$ and time $t$, using IPTW $\hat{w}_i$ and pseudo-observations $\hat{\theta}_{Zi}$.

The paper entails a simulation study in which the authors perform the methods above. Scenarios were created with synthetic datasets and predefined covariates and outcomes. ATE was used to estimate efficacy with the presence of non-proportionality. Findings indicate that non-parametric models encapsulated lower bias regarding outcomes, with the cost of increased standard errors compared with parametric and semi parametric models. The authors indicate that no single method excelled in all scenarios. The authors explore the two “real-world” scenarios in cancer therapies, sarcoma dataset from the National Cancer database(NCDB) and renal cancer. This is to confirm the simulation studies. The paper concludes, indicating the considerations for implementing the methods explored in the paper with practicality and computational efficiency in mind, the authors recommend IPTW KM due to its simplicity and robustness.
}
}
%%

@misc{smith_scoping_2022,
title = {A scoping methodological review of simulation studies comparing statistical and machine learning approaches to risk prediction for time-to-event data},
url = {https://doi.org/10.1186/s41512-022-00124-y},
journal ={Diagnostic and Prognostic Research},
author = {Smith, Hayley and Sweeting, Michael and Morris, Tim and Crowther, Michael J.},
date = {2022},
annote = {\textbf{Aim:}
Qualitative analysis, providing trends and gaps found within current methodologies and literature in survival analysis research.
\\\\
\textbf{Style/Type:}
journal article, empirical
\\\\
\textbf{Cross references:}
The information presented outline, key considerations even formal standards to apply when undertaking comparitive study regarding survival analysis methods.
\\\\
\textbf{Summary:}
The paper explores the Pubmed database, regarding simulations studies on time to event data, where both statistical and machine learning approaches for risk prediction are used. The first part entails the mechanism the authors utilised to derive a target set of 10 papers most relevant to criterion for selection. The comparative findings are broken into categories for analysis, where components are evaluated within several sections. The study looks at the data generating methods (DGM) used in the models, and the coverage these mechanisms obtain in terms of sample sizes, covariate attributes, censoring and failure distributions to name a few. Then covariates and failure time analysis within the studies, are compared across data distribution families and covariate relationships and effects and operational model assumptions. Next percentage censoring is compared across underlying distribution families and training subsets splits with regards to DGM methods used. The studies are categorised into three sets noting which statistical and machine learning models the studies used namely statistical methods, which include cox proportional hazards and variants thereof, hybrid models like the superlearner method and Mahalanobis K nearest neighbour, and machine learning methods like random survival forests and support vector machines. Finally the studies are compared across estimands like survival function, hazard functions and linear predictors to name a few and performance measures, which entails common methods like the C-index, mean square prognostic error etc. Findings of the paper point to key considerations, when conducting research simulation studies, of the nature of the underlying papers. Authors highlight poor reporting standards across studies, as well as missing information on the implementation of DGM’s where it is noted that the method performance is directly dependent on the DGM. Furthermore it is pointed out that there is a level of bias towards machine learning methods used. The authors suggest that researchers consider comprehensive evaluation, and the use of standards, pointing to two sources of interest, and highlight the need for simulation studies that perform method comparisons independent of novel method development, assessing both discrimination and calibration, report variations in performance measures and consider fairness for comparing underlying methods.
}
}
%%
