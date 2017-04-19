---
output: html_document
---
## Unrealistic optimism in advice-taking: A computational account 
The online supplement for our paper <i> Unrealistic optimism in advice-taking: A computational account </i> is hosted on GitHub at [https://github.com/ycleong/AdviceTaking](https://github.com/ycleong/AdviceTaking). In service of other researchers who would like to reproduce our work or are interested in applying the same models in their own work, we have made our data and code available. If you have any questions, or find any bugs, please email me at ycleong@stanford.edu.

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
* [Bayesian Learning Model](code/models/NoCB_Learner.m)
* [Confirmatiom Bias Model](code/models/NewCB_Learner.m)
* [Choice Model (Softmax function)](code/models/basic_bayes.m)
* Find best fit value for alpha and beta for prior distribution
    * [Version 1](code/expt1/findpriors.m): One prior for all advisors
    * [Version 2](code/expt2/findpriors_expt2.m): Different prior for each advisor
* Simulating the model performing the Advisor Evaluation Phase: 
    * [Expt 1](code/expt1/sim_model.m)  
    * [Expt 2](code/expt2/sim_model_expt2.m)
* [Fitting participants' performance on the stock prediction task](code/StockPrediction.m)

### Example Face Stimuli
Face stimuli were drawn from the Interdisciplinary Affective Science Laboratory (IASLab) Face Set. We used faces posing calm expressions with mouth closed and eyes gazing straight ahead. We added 6 example faces here as supplemental material for our manuscript. Users who wish to use this dataset should contact the owners at www.affective-science.org. If requested by the owners, we would be happy to remove the 6 example photos here.
* [Example Stimuli](stimuli/)