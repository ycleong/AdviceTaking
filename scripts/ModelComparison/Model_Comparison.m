%% Model Comparison Script
% This script loads model fits and compares models based on corrected average likelihood
%     - Requires that you already run the model fits from each model
%     (Fits_[modelname]_[experimentno].mat)
%     - Also runs Bayesian Estimation to examine if within-participant
%     differences in corrected average likelihood per trial of the first two models are credibly
%     different from zero
%        - Dependencies: 
%             - Nils Winter's Matlab Toolbox for Bayesian Estimation, available
%               at: https://github.com/NilsWinter/matlab-bayesian-estimation
%             - JAGS: http://mcmc-jags.sourceforge.net/  
%             - matjags: http://psiexp.ss.uci.edu/research/programs_data/jags/
%        - To Run Bayesian estimation, set run_bayesian_estimation to 1 (line 20)
% 
% YC Leong 7/23/2017

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Load Models                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear mex
clear all

run_bayesian_estimation = 0;

% Set Directories and load data
dirs.results = '../../results';
experiment = 'Expt1';

% Which models to run? ['CB','NoCB','WSLS','RL','Null']
%   CB = Confirmation Bias
%   NoCB = Bayesian Learning
%   WSLS = Win-stay-lose-shift
%   RL = Reinforcement Learning
%   Null = Baseline model

% models = {'CB','NoCB','WSLS','RL','Null'};
models = {'CB','NoCB'};

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Load Models                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load models and save AIC
for m = 1:length(models)
    Model{m,1} = load(fullfile(dirs.results,sprintf('Fits_%s_%s.mat',models{m},experiment)));
    AIC(:,m) = Model{m}.AIC;
end

nSub = length(AIC);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Model Comparison Table                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate model comparison metric
for m = 1:length(models)
    lik{m,1} = sprintf('%0.1f(%0.1f)',mean(Model{m,1}.lik),std(Model{m,1}.lik)/sqrt(nSub));
    AIC_mean{m,1} = sprintf('%0.1f(%0.1f)',mean(Model{m,1}.AIC),std(Model{m,1}.AIC)/sqrt(nSub));
    AvgLik{m,1} = sprintf('%0.2f(%0.2f)',mean(Model{m,1}.gm_AIC),std(Model{m,1}.gm_AIC)/sqrt(nSub));
    all_gm_AIC(:,m) = Model{m,1}.gm_AIC;
    
end

% Number of best-fit participants
[max_aic, max_m] = max(all_gm_AIC'); 
for m = 1:length(models)
    best_model(m,:) = sum(max_m == m);
end

T = table(lik,AIC_mean,AvgLik,best_model,'RowNames',models)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Bayesian Estimation                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if run_bayesian_estimation
    y = all_gm_AIC(:,1) - all_gm_AIC(:,2);
    mbe_1gr(y,0);
end


