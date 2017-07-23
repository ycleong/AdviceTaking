%% Wrapper script to fit Bayesian and Confirmation Bias model to Experiment 1 data
% This script performs maximum a posteri fitting to find best-fit values for each subject. 
% Finds MAP estimate for alpha, beta and tau for each participant
% 
% YC Leong 7/21/2017

clear mex
clear all

% Which model to fit: 'Bayesian or Confirmation Bias'
Fit.Model = 'ConfirmationBias';

% Run fitting procedure, or proceed from intermediate results?
run_fit = 1;

% Set Directories and load data
dirs.data = '../../data';
dirs.results = '../../results';
load(fullfile(dirs.data,'AllData.mat'));

% Add helper scripts
addpath(genpath('../../scripts'));

% Subjects to fit
Sub = [101 102 103 104 105 106 107 108 109 110 112 113 114 115 116 118 119 ...
    120 121 122 123 124 125 126 127 128];
nSub = length(Sub);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Initialize parameters for model-fitting                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fit.Subjects = Sub; % which subjects to fit
Fit.NIter = 3;      % how many iterations to fit

% Initialize paremters to fit: alpha, beta, tau
Fit.Nparms = 3; 
Fit.LB = [0.01 0.01 1e-16];
Fit.UB = [20 20 100];

Fit.Start = ones(1,length(Fit.Subjects));  
Fit.End = ones(1,length(Fit.Subjects))*108;

% Add regulatory gamma priors?
% Alpha
Fit.Priors.Use(1) = 1;  
Fit.Priors.Parms(1,1) = 2;
Fit.Priors.Parms(1,2) = 3;

% Beta
Fit.Priors.Use(2) = 1;  
Fit.Priors.Parms(2,1) = 2;
Fit.Priors.Parms(2,2) = 3;

% Tau
Fit.Priors.Use(3) = 1;   
Fit.Priors.Parms(3,1) = 2;
Fit.Priors.Parms(3,2) = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Fit Model                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if run_fit
    
    % Iterate over participants
    for s = 1:nSub
        thisData = AllData{s,3}.Learn{1,1};
        
        fprintf('Subject %i \n',Sub(s));
        
        % Get Advisor Correct and Choice
        for j = 1:3
            AdvisorCorrect(:,j) = thisData.AdvisorCorrect(find(thisData.Advisor == j));
            Choice(:,j) = thisData.Choice(find(thisData.Advisor == j));
        end
        
        % Get Choice
        choice = [Choice(:,1); Choice(:,2); Choice(:,3)];

        Fit.NTrials(s) = sum(thisData.ValidTrials);
        nTrials = length(thisData.ValidTrials);
        
        % Fit model 
        for iter = 1:Fit.NIter
            Fit.init(s,iter,[1]) = rand*5;
            Fit.init(s,iter,[2]) = rand*5;
            Fit.init(s,iter,[3]) = rand*5;
            
            [res,lik,flag,out,lambda,grad,hess] = ...
                fmincon(@(x) jointfit_model(AdvisorCorrect,choice, Fit.Priors,x,Fit.Model),...
                Fit.init(s,iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
                'GradObj','off','DerivativeCheck','off','LargeScale','off','Algorithm','active-set','Hessian','off','Display','on'));
            
            Fit.Result.h1(s,:,iter) = res(1);
            Fit.Result.t1(s,:,iter) = res(2);
            Fit.Result.tau(s,:,iter) = res(3);
            
            Fit.Result.Hessian(s,:,:,iter) = full(hess); % save the Hessian to do Laplace approx later; it was sparse initially, so use "full" to expand
            Fit.Result.Lik(s,iter) = lik;
            
            fprintf('h1 = %0.3f, t1 = %0.3f, tau = %0.3f \n',res(1),res(2),res(3));
            
        end
        
    end
    
    % Find best fit parameters
    [a,b] = min(Fit.Result.Lik,[],2);
    
    for s = 1:length(Fit.Subjects)
        Fit.Result.BestFit(s,:) = [Fit.Subjects(s),...
            Fit.Result.h1(s,b(s)),...
            Fit.Result.t1(s,b(s)),...
            Fit.Result.tau(s,b(s)),...
            Fit.Result.Lik(s,b(s))];
    end
    
    % Save data
    switch Fit.Model
        case 'ConfirmationBias'
            save_file = fullfile(dirs.results,sprintf('Fits_CB_Expt1.mat'));
        case 'Bayesian'
            save_file = fullfile(dirs.results,sprintf('Fits_NoCB_Expt1.mat'));
    end
    
    save (save_file,'Fit');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Run model with best fit parameters                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch Fit.Model
    case 'ConfirmationBias'
        save_file = fullfile(dirs.results,sprintf('Fits_CB_Expt1.mat'));
    case 'Bayesian'
        save_file = fullfile(dirs.results,sprintf('Fits_NoCB_Expt1.mat'));
end
load(save_file)

% Iterate over participants
for s = 1:nSub
    
    fprintf('Subject %i \n',Sub(s));
    
    thisData = AllData{s,3}.Learn{1,1};
    
    % Get Advisor Correct and Choice
    for j = 1:3
        AdvisorCorrect(:,j) = thisData.AdvisorCorrect(find(thisData.Advisor == j));
        Choice(:,j) = thisData.Choice(find(thisData.Advisor == j));
    end
    
    % Get Choice
    choice = [Choice(:,1); Choice(:,2); Choice(:,3)];
    nTrials = length(thisData.ValidTrials);

    x(1) = Fit.Result.BestFit(s,2);
    x(2) = Fit.Result.BestFit(s,3);
    x(3) = Fit.Result.BestFit(s,4);
    
    % Switch off regulatory priors
    Fit.Priors.Use(1) = 0;
    Fit.Priors.Use(2) = 0; 
    Fit.Priors.Use(3) = 0; 
    
    [lik(s,1) latents{s,1}] = jointfit_model(AdvisorCorrect,choice, Fit.Priors,x,Fit.Model);
    
    % Calculate AIC and BIC
    AIC(s,1) = 2*lik(s) + Fit.Nparms*2 + (2*Fit.Nparms*(Fit.Nparms+1))/(Fit.NTrials(s)-Fit.Nparms-1);
    BIC(s,1) = 2*lik(s) + Fit.Nparms * log(Fit.NTrials(s)); 
end

% Calculate average likelihood per trial
g_mean(:,1) = exp(-lik ./ Fit.NTrials');
% Calculate corrected average likelihood per trial based on AIC
gm_AIC(:,1) = exp(-0.5 .* AIC ./ Fit.NTrials');
% Calculate corrected average likelihood per trial based on BIC
gm_BIC(:,1) = exp(-0.5 .* BIC ./ Fit.NTrials');

% Save data
switch Fit.Model
    case 'ConfirmationBias'
        save_file = fullfile(dirs.results,sprintf('Fits_CB_Expt1.mat'));
    case 'Bayesian'
        save_file = fullfile(dirs.results,sprintf('Fits_NoCB_Expt1.mat'));
end

% Print out results
fprintf('log lik = %0.1f(%0.1f), AIC = %0.1f(%0.1f), avg lik = %0.2f \n',...
    mean(lik),std(lik)/sqrt(nSub),mean(AIC),std(AIC)/sqrt(nSub),mean(gm_AIC)); 

% Save all files
save (save_file,'Fit','lik','latents','g_mean','gm_AIC','gm_BIC','AIC','BIC');