---
output: html_document
---
## Unrealistic optimism in advice-taking: A computational account 
The online supplement for our paper <i> Unrealistic optimism in advice-taking: A computational account </i> is hosted on GitHub at [https://github.com/ycleong/AdviceTaking](https://github.com/ycleong/AdviceTaking). In service of other researchers who would like to reproduce our work or are interested in applying the same models in their own work, we have made our data and code available. If you have any questions, or find any bugs (or broken links), please email me at ycleong@stanford.edu.

Click [here](summary.md) for a summary of our findings

### Data
Data from studies 1 and 2 are Matlab data structures that can be downloaded here:  
  - [Study 1](data/AllData.mat)  
  - [Study 2](data/AllData_Expt2.mat)

Data from Study 3 are saved in a series of CSV files that can be downloaded here:  

- Generation 1: [Stock Prediction](data/dPrivate_gen1.csv), [Advisor Evaluation](data/dSocial_gen1.csv), [Joint Prediction](data/dJoint_gen1.csv)  
- Generation 2: [Stock Prediction](data/dPrivate_gen2.csv), [Advisor Evaluation](data/dSocial_gen2.csv), [Joint Prediction](data/dJoint_gen2.csv)  

See [here](data/readme_exptfiles.md) for a readme for information about the important variables.  


### Code
#### Modeling the Advisor Evaluation Phase
* [Confirmation Bias Model](scripts/CB_BL/NewCB_Learner.m)  
* [Bayesian Learning Model](scripts/CB_BL/NoCB_Learner.m)  
* [Win-Stay-Lose-Shift Model](scripts/WSLS/WSLS_lik.m)  
* [Reinforcement Learning Model](scripts/RL/RL_lik.m)  

Wrapper scripts for all models can be found in the same folder, [modelname]_wrapper_expt[experiment number].m. For example, to run the Confirmation Bias or Bayesian Learning models for Experiment 1 data, go to [scripts/CB_BL](scripts/CB_BL) and run CB_BL_wrapper_expt1.m. 

#### Model Comparison
[ModelComparison.m](scripts/ModelComparison/ModelComparison.m): Loads model fits and compares models based on corrected average likelihood per trial.  

- Requires that you already fit each model you want to compare (see above)  
- Also runs Bayesian Estimation to examine if within-participant differences in corrected average likelihood per trial of the first two models are credibly different from zero  
   - Dependencies:
       - Nils Winter's Matlab Toolbox for Bayesian Estimation, available at: [https://github.com/NilsWinter/matlab-bayesian-estimation](https://github.com/NilsWinter/matlab-bayesian-estimation)
       - JAGS: [http://mcmc-jags.sourceforge.net/](http://mcmc-jags.sourceforge.net/)  
       - matjags: [http://psiexp.ss.uci.edu/research/programs_data/jags/](http://psiexp.ss.uci.edu/research/programs_data/jags/)  

#### Modeling the stock trend  
* [Fitting Bayesian Learning model to the stock trend](scripts/StockPrediction/StockPrediction.m)  
  
  
### Example Face Stimuli
Face stimuli were drawn from the Interdisciplinary Affective Science Laboratory (IASLab) Face Set. We used male caucasian faces faces posing calm expressions with mouth closed and eyes gazing straight ahead. We added 6 example faces here as supplemental material for our manuscript. Users who wish to use this dataset should contact the owners at www.affective-science.org. If requested by the owners, we would be happy to remove the 6 example photos here.  

* [Example Stimuli](stimuli/)