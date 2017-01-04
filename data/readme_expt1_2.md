---
output: html_document
---
## Readme for data structures from Study 1 and 2
* AllData{:,1}: Subject IDs  
* AllData{:,2}: Task Parameters
* AllData{:,3}: Task Data  
    * NoAdvice: Data from Stock Prediction Phase
        * Choice: Participants' prediction (1 = UP, 0 = DOWN)
        * RT: Reaction Time
        * StockOutcome: Outcome of Stock (1 = UP, -1 = DOWN)
        * PlayerOutcome: Whether the participant guessed the price change correctly (1 = correct, -1 = incorrect)
        * pUP: Probability that the stock will increase in price
    * Learn: Data from Advisor Evaluation Phase  
        * Choice: Participants' bet (1 = FOR, 0 = AGAINST)  
        * RT: Reaction Time 
        * Advisor: Which advisor?
            * Expt 1: 1 = 75%, 2 = 50%, 3 = 25%
            * Expt 2: 1 = 1-star 25%, 2 = 2-star 50%, 3 = 3-star 50%, 4 = 4-star 75%  
        * AdvisorPred: Advisor's prediction (1 = UP, -1 = DOWN)
        * AdvisorCorrect: Whether the advisor was correct (1 = correct, -1=incorrect)
        * PlayerOutcome: Whether the participants' bet was correct (1 = correct, -1 = incorrect)
        * StockOutcome = Outcome of Stock (1 = UP, -1 = DOWN)
        * pUP = Probability that the stock will increase in price  
    * Advice = Data from Advisor Evaluation Phase  
        * Choice: Participants' prediction (1 = UP, 0 = DOWN)  
        * RT: Reaction Time 
        * Advisor: Which advisor?
            * Expt 1: 1 = 75%, 2 = 50%, 3 = 25%
            * Expt 2: 1 = 1-star 25%, 2 = 2-star 50%, 3 = 3-star 50%, 4 = 4-star 75%  
        * AdvisorPred: Advisor's prediction (1 = UP, -1 = DOWN)
        * AdvisorCorrect: Whether the advisor was correct (1 = correct, -1=incorrect)
        * PlayerOutcome: Whether the participants' prediction was correct (1 = correct, -1 = incorrect)
        * StockOutcome = Outcome of Stock (1 = UP, -1 = DOWN)
        * pUP = Probability that the stock will increase in price  
* AllData{:,4} = Data from 10 practice trials  


