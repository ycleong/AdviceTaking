%% Model-fitting script for Experiment 2
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

function [lik,latents] = jointfit_model_expt2(AdvisorCorrect,choice,FitParms,X,this_model)

% Initialize Parameters
h1 = X(1);
t1 = X(2);
h2 = X(3);
t2 = X(4);
h3 = X(5);
t3 = X(6);
h4 = X(7);
t4 = X(8);
tau = X(9);
cb = 1;
lik = 0;
latents.pUP = NaN(length(choice),1);
latents.choice_prob = NaN(length(choice),1);

% Initialize Prior
x = [0.01:0.01:0.99];
betaprior1 = betapdf(x,h1,t1);
betaprior2 = betapdf(x,h2,t2);
betaprior3 = betapdf(x,h3,t3);
betaprior4 = betapdf(x,h4,t4);

% Which model to run ?
switch this_model
    
    % Fit Confirmation Bias and Bayesian Learners
    case 'ConfirmationBias'
        A1 = NewCB_Learner(AdvisorCorrect(:,1),betaprior1',cb);
        A2 = NewCB_Learner(AdvisorCorrect(:,2),betaprior2',cb);
        A3 = NewCB_Learner(AdvisorCorrect(:,3),betaprior3',cb);
        A4 = NewCB_Learner(AdvisorCorrect(:,4),betaprior4',cb);
        
    case 'Bayesian'
        A1 = NoCB_Learner(AdvisorCorrect(:,1),betaprior1',cb);
        A2 = NoCB_Learner(AdvisorCorrect(:,2),betaprior2',cb);
        A3 = NoCB_Learner(AdvisorCorrect(:,3),betaprior3',cb);
        A4 = NoCB_Learner(AdvisorCorrect(:,4),betaprior4',cb);
end

% Concatentate trials 
p_dist = [A1.p_dist(1:end-1,:); A2.p_dist(1:end-1,:); A3.p_dist(1:end-1,:); A4.p_dist(1:end-1,:)];
pHat = [A1.pUP(1:end-1); A2.pUP(1:end-1); A3.pUP(1:end-1); A4.pUP(1:end-1)];
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
for i = 1:9
    if (FitParms.Use(i))    
        lik = lik - log(gampdf(X(i),FitParms.Parms(i,1),FitParms.Parms(i,2)));
    end
end

end