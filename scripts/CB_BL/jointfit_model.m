%% Model-fitting script for Experiment 1 and Experiment 1 Generation 1
% Inputs -
%     AdvisorCorrect : vector of [1,0] indicating history of advisor performance
%     choice: prior distribution 
%     FitParms: Paraeters for regulatory priors
%     X: Vector storing parameters to fit
%     this_model: which model to fit? 'ConfirmationBias' or Bayesian'
% Outputs -
%     lik - negative log likelihood
%     latents - latent variables: 
%         pUP: probability of betting for an advisor
%         choice_prob: choice probability on that trial
% 
% YC Leong 7/21/2017

function [lik,latents] = jointfit_model(AdvisorCorrect,choice,FitParms,X,this_model)

% Initialize Parameters
h1 = X(1);  % alpha
t1 = X(2);  % beta
tau = X(3); % tau
cb = 1;     % degree of confirmation bias
lik = 0;    % log likelihood
latents.pUP = NaN(length(choice),1); % probability of betting for
latents.choice_prob = NaN(length(choice),1); % choice probability

% Initialize Prior
x = [0.01:0.01:0.99];
y1 = betapdf(x,h1,t1);
betaprior = y1;


% Which model to run ?
switch this_model
    
    % Fit Confirmation Bias and Bayesian Learners
    case 'ConfirmationBias'
        A1 = NewCB_Learner(AdvisorCorrect(:,1),betaprior',cb);
        A2 = NewCB_Learner(AdvisorCorrect(:,2),betaprior',cb);
        A3 = NewCB_Learner(AdvisorCorrect(:,3),betaprior',cb);
        
    case 'Bayesian'
        A1 = NoCB_Learner(AdvisorCorrect(:,1),betaprior',cb);
        A2 = NoCB_Learner(AdvisorCorrect(:,2),betaprior',cb);
        A3 = NoCB_Learner(AdvisorCorrect(:,3),betaprior',cb);
   
end

% Concatentate trials 
p_dist = [A1.p_dist(1:end-1,:); A2.p_dist(1:end-1,:); A3.p_dist(1:end-1,:)];
pHat = [A1.pUP(1:end-1); A2.pUP(1:end-1); A3.pUP(1:end-1)];
pHat(find(isnan(pHat))) = pHat(find(isnan(pHat))-1);

% Fit choice probabilities (tau)
for t = 1:length(choice)
    if ~isnan(choice(t))
        pUP = 1/(1+exp(-tau*(pHat(t)-0.5)));
        
        if choice(t)
            choice_prob = pUP;
        else
            choice_prob = 1-pUP;
        end
        
        lik = lik + log(choice_prob);
    
        latents.pUP(t) = pUP;
        latents.choice_prob(t) = choice_prob;
    end
end

% calculate negative loglikelihood 
lik = -lik;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     Add regulatory priors                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1
    
    % Add prior on alpha
    if (FitParms.Use(1))
        lik = lik - log(gampdf(X(1),FitParms.Parms(1,1),FitParms.Parms(1,2)));
    end
    
    % Add prior on beta
    if (FitParms.Use(2))
        %lik = lik - log(exppdf(X(2),FitParms.Parms(2,1)));
        lik = lik - log(gampdf(X(2),FitParms.Parms(2,1),FitParms.Parms(2,2)));
    end
    
    % Add prior on tau
    if (FitParms.Use(3))    % putting a Gamma prior on Beta
        lik = lik - log(gampdf(X(3),FitParms.Parms(3,1),FitParms.Parms(3,2)));
    end
end

end