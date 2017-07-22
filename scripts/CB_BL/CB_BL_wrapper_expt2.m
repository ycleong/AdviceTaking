%% Wrapper script to fit Bayesian and Confirmation Bias model to Experiment 2 data
% This script performs maximum a posteri fitting to find best-fit values for each subject. 
% Finds MAP estimate for alpha, beta and tau for each participant
% 
% YC Leong 7/21/2017

clear mex
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    Set up                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Which model to fit: 'Bayesian or Confirmation Bias'
Model = 'Bayesian';

run_fit = 1;

% Set Directories and load data
dirs.data = '../../data';
dirs.results = '../../results';
load(fullfile(dirs.data,'AllData_Expt2.mat'));

% Add helper scripts
addpath('../../scripts');

% Subjects to fit
Sub = [101:130];
nSub = length(Sub);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Initialize parameters for model-fitting                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fit.Subjects = Sub; % which subjects to fit
Fit.Model = Model;  % which model to fit
Fit.NIter = 1;      % how many iterations of fits to run

Fit.Start = ones(1,length(Fit.Subjects));
Fit.End = ones(1,length(Fit.Subjects))*108;

% Initialize paremters to fit: alpha, beta for each advisor, tau
Fit.Nparms = 9; % Alpha Beta Tau
Fit.LB = [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 1e-16];
Fit.UB = [20 20 20 20 20 20 20 20 100];

% Add regulatory gamma priors?
Fit.Priors.Use(1) = 1;   
Fit.Priors.Parms(1,1) = 2;
Fit.Priors.Parms(1,2) = 3;

Fit.Priors.Use(2) = 1;   
Fit.Priors.Parms(2,1) = 2;
Fit.Priors.Parms(2,2) = 3;

Fit.Priors.Use(3) = 1;   
Fit.Priors.Parms(3,1) = 2;
Fit.Priors.Parms(3,2) = 3;

Fit.Priors.Use(4) = 1;   
Fit.Priors.Parms(4,1) = 2;
Fit.Priors.Parms(4,2) = 3;

Fit.Priors.Use(5) = 1;   
Fit.Priors.Parms(5,1) = 2;
Fit.Priors.Parms(5,2) = 3;

Fit.Priors.Use(6) = 1;   
Fit.Priors.Parms(6,1) = 2;
Fit.Priors.Parms(6,2) = 3;

Fit.Priors.Use(7) = 1;   
Fit.Priors.Parms(7,1) = 2;
Fit.Priors.Parms(7,2) = 3;

Fit.Priors.Use(8) = 1;   
Fit.Priors.Parms(8,1) = 2;
Fit.Priors.Parms(8,2) = 3;

Fit.Priors.Use(9) = 1;  
Fit.Priors.Parms(9,1) = 2;
Fit.Priors.Parms(9,2) = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Fit Model                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if run_fit
    
    % Iterate over participants
    for s = 1:length(Fit.Subjects)
        thisData = AllData{s,3}.Learn{1,1};
        fprintf('Subject %i \n',Sub(s))
        
        % Get Advisor Correct
        for j = 1:4
            AdvisorCorrect(:,j) = thisData.AdvisorCorrect(find(thisData.Advisor == j));
            Choice(:,j) = thisData.Choice(find(thisData.Advisor == j));
        end
        
        choice = [Choice(:,1); Choice(:,2); Choice(:,3); Choice(:,4)];
        
        Fit.NTrials(s) = sum(thisData.ValidTrials);
        nTrials = length(thisData.ValidTrials);
        
        % Fit model 
        for iter = 1:Fit.NIter
            
            Fit.init(s,iter,[1]) = rand*5;
            Fit.init(s,iter,[2]) = rand*5;
            Fit.init(s,iter,[3]) = rand*5;
            Fit.init(s,iter,[4]) = rand*5;
            Fit.init(s,iter,[5]) = rand*5;
            Fit.init(s,iter,[6]) = rand*5;
            Fit.init(s,iter,[7]) = rand*5;
            Fit.init(s,iter,[8]) = rand*5;
            Fit.init(s,iter,[9]) = rand*5;
            
            [res,lik,flag,out,lambda,grad,hess] = ...
                fmincon(@(x) jointfit_model_expt2(AdvisorCorrect,choice, Fit.Priors,x,Fit.Model),...
                Fit.init(s,iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
                'GradObj','off','DerivativeCheck','off','LargeScale','off','Algorithm','active-set','Hessian','off','Display','on'));
            
            Fit.Result.h1(s,:,iter) = res(1);
            Fit.Result.t1(s,:,iter) = res(2);
            Fit.Result.h2(s,:,iter) = res(3);
            Fit.Result.t2(s,:,iter) = res(4);
            Fit.Result.h3(s,:,iter) = res(5);
            Fit.Result.t3(s,:,iter) = res(6);
            Fit.Result.h4(s,:,iter) = res(7);
            Fit.Result.t4(s,:,iter) = res(8);
            Fit.Result.tau(s,:,iter) = res(9);
            
            Fit.Result.Hessian(s,:,:,iter) = full(hess); % save the Hessian to do Laplace approx later; it was sparse initially, so use "full" to expand
            Fit.Result.Lik(s,iter) = lik;
            
            fprintf('[%0.3f %0.3f, %0.3f %0.3f, %0.3f %0.3f, %0.3f %0.3f] tau = %0.3f \n',...
                res(1),res(2),res(3),res(4),res(5),res(6),res(7),res(8),res(9));
            
        end
        
    end
    
    % Find best fit parameters
    [a,b] = min(Fit.Result.Lik,[],2);
    
    for s = 1:length(Fit.Subjects)
        Fit.Result.BestFit(s,:) = [Fit.Subjects(s),...
            Fit.Result.h1(s,b(s)),...
            Fit.Result.t1(s,b(s)),...
            Fit.Result.h2(s,b(s)),...
            Fit.Result.t2(s,b(s)),...
            Fit.Result.h3(s,b(s)),...
            Fit.Result.t3(s,b(s)),...
            Fit.Result.h4(s,b(s)),...
            Fit.Result.t4(s,b(s)),...
            Fit.Result.tau(s,b(s)),...
            Fit.Result.Lik(s,b(s))];
    end
    
    % Save data
    switch Fit.Model
        case 'ConfirmationBias'
            save_file = fullfile(dirs.results,sprintf('Fits_CB_Expt2.mat'));
        case 'Bayesian'
            save_file = fullfile(dirs.results,sprintf('Fits_NoCB_Expt2.mat'));
    end
    
    save (save_file,'Fit');    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Run model with best fit parameters                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch Fit.Model
    case 'ConfirmationBias'
        save_file = fullfile(dirs.results,sprintf('Fits_CB_Expt2.mat'));
    case 'Bayesian'
        save_file = fullfile(dirs.results,sprintf('Fits_NoCB_Expt2.mat'));
end
load(save_file)

for s = 1:nSub
    
    fprintf('Subj %i \n',Sub(s));
    
    % Iterate over participants
    thisData = AllData{s,3}.Learn{1,1};
    
    % Get Advisor Correct and Choice
    for j = 1:4
        AdvisorCorrect(:,j) = thisData.AdvisorCorrect(find(thisData.Advisor == j));
        Choice(:,j) = thisData.Choice(find(thisData.Advisor == j));
    end
    
    % Get Choice
    choice = [Choice(:,1); Choice(:,2); Choice(:,3); Choice(:,4)];
    
    nTrials = length(thisData.ValidTrials);
    
    x(1) = Fit.Result.BestFit(s,2);
    x(2) = Fit.Result.BestFit(s,3);
    x(3) = Fit.Result.BestFit(s,4);
    x(4) = Fit.Result.BestFit(s,5);
    x(5) = Fit.Result.BestFit(s,6);
    x(6) = Fit.Result.BestFit(s,7);
    x(7) = Fit.Result.BestFit(s,8);
    x(8) = Fit.Result.BestFit(s,9);
    x(9) = Fit.Result.BestFit(s,10);
    
    % Switch off regulatory priors
    Fit.Priors.Use(:) = 0;
    
    [lik(s,1) latents{s,1}] = jointfit_model_expt2(AdvisorCorrect,choice, Fit.Priors,x,Fit.Model);
    
    % Calculate AIC and BIC
    AIC(s,1) = 2*lik(s) + Fit.Nparms*2 + (2*Fit.Nparms*(Fit.Nparms+1))/(Fit.NTrials(s)-Fit.Nparms-1);
    BIC(s,1) = 2*lik(s) + Fit.Nparms * log(Fit.NTrials(s)); 
end

% Calculate average likelihood per trial
g_mean(:,1) = exp(-lik ./ Fit.NTrials');
% Calculate corrected average likelihood per trial based on AIC
gm_AIC(:,1) = exp(-0.5 .* AIC ./ Fit.NTrials');
% Calculate average likelihood per trial based on BIC
gm_BIC(:,1) = exp(-0.5 .* BIC ./ Fit.NTrials');

% Save data
switch Fit.Model
    case 'ConfirmationBias'
        save_file = fullfile(dirs.results,sprintf('Fits_CB_Expt2.mat'));
    case 'Bayesian'
        save_file = fullfile(dirs.results,sprintf('Fits_NoCB_Expt2.mat'));
end

% Print out results
fprintf('log lik = %0.1f(%0.1f), AIC = %0.1f(%0.1f), avg lik = %0.2f \n',...
    mean(lik),std(lik)/sqrt(nSub),mean(AIC),std(AIC)/sqrt(nSub),mean(gm_AIC)); 

% Save all files
save (save_file,'Fit','lik','latents','g_mean','gm_AIC','gm_BIC','AIC','BIC');


