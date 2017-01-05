---
output: html_document
---
## Unrealistic optimism in advice-taking: A computational account 
The online supplement for our paper <i> Unrealistic optimism in advice-taking: A computational account </i> is hosted on GitHub at [https://github.com/ycleong/AdviceTaking](https://github.com/ycleong/AdviceTaking). In service of other researchers who would like to reproduce our work or are interested in applying the same models in their own work, we have made our data and code available. If you have any questions, or find any bugs, please email me at ycleong@stanford.edu.

### Summary
Across a variety of decision-making domains, “expert” advisors are at chance at making accurate predictions. Yet, individuals continue to take their advice seriously, often at a significant personal cost. For example, investors are influenced by the advice of financial gurus, even when these gurus are no better than chance with their market forecasts. Similarly, politicians consult political pundits before making decisions, even though these pundits fail to accurately predict public opinion. Why do individuals have inflated beliefs about advisors’ expertise? More intriguingly, why do these inflated beliefs persist despite repeated experience with the advisors?

We propose that unrealistic optimism in advisors arise from a combination of optimistic initial expectations and confirmation bias in how these expectations are updated in light of new information. Across three studies, participants performed a financial decision-making task where they learned about the expertise of different financial advisors and made decisions about utilizing each advisor’s advice when making stock predictions. Participants in our task were indeed overly optimistic about advisors’ expertise, and utilized advice more than warranted by the advisors’ objective accuracy at predicting the stock. Participants’ beliefs about the advisor’s expertise were best tracked by the trial-by-trial predictions made by a computational model that assumes optimistic initial expectations and preferentially incorporates expectation-consistent information. By fitting the models to participants’ behavior, we arrived at quantitative and precise estimates of participants’ initial expectations about the advisor. In Study 2, we experimentally manipulated these expectations and corrected for the optimism bias, as predicted by our computational models. In Study 3, we investigated the use of crowd-sourced ratings to calibrate participants’ initial expectations, but demonstrated instead that the ratings propagated and exaggerated the optimism bias.

The mechanisms underlying the optimistic bias observed in the current set of studies are basic cognitive processes that are likely to impact advice-taking behavior beyond financial decision-making scenarios. From healthcare professionals to political pundits, policy advisors to sports commentators, advisors are often portrayed as experts in their respective fields. Decision-makers are likely to have optimistic expectations about these advisors, expectations that could be wrong yet resistant to change. We believe that this research highlights the importance of tabulating and making public quantitative metrics of advisor accuracy, such that decision-makers can consider them when deciding whether to utilize a piece of advice. Advisors are often helpful, but knowing when they are not can help decision-makers discern how to incorporate advice when making choices.

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