%% Wrapper script to fit Bayesian and Confirmation Bias model to Experiment 3 Generation 1 data
% This script performs maximum a posteri fitting to find best-fit values for each subject. 
% Finds MAP estimate for alpha, beta and tau for each participant
% YC Leong 7/21/2017

clear mex
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    Set up                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Which model to fit: 'Bayesian or Confirmation Bias'
Fit.Model = 'Bayesian';

% Run fitting procedure, or proceed from intermediate results?
run_fit = 1;

% Set Directories and load data
dirs.data = '../../data';
dirs.results = '../../results';
addpath(genpath('../../scripts'));

% Subjects
dSocial = csvread(fullfile(dirs.data,'dSocial_gen1.csv'),1,1);
Sub = unique(dSocial(:,1));
nSub = length(Sub);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Initialize parameters for model-fitting                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set fitting parameters
Fit.Subjects = Sub;
Fit.NIter = 1;
Fit.Start = ones(1,length(Fit.Subjects));
Fit.End = ones(1,length(Fit.Subjects))*108;

Fit.Nparms = 3; % Alpha Beta Tau
Fit.LB = [0.01 0.01 1e-16];
Fit.UB = [20 20 100];

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Fit Model                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if run_fit
for s = 1:nSub
    thisData = dSocial(dSocial(:,1) == Sub(s),:);
    
    fprintf('Subject %i \n',s);
    
    % Get Advisor Correct and Choice
    for j = 1:3
        AdvisorCorrect(:,j) = thisData(thisData(:,5) == j,6);
        Choice(:,j) = thisData(thisData(:,5) == j,7);
    end
    choice = [Choice(:,1); Choice(:,2); Choice(:,3)];
    
    Fit.NTrials(s) = length(thisData);

    % Fit model 3 times
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


%% Save data
switch Fit.Model
    case 'ConfirmationBias'
        save_file = fullfile(dirs.results,sprintf('Fits_CB_Expt3_Gen1.mat'));
    case 'Bayesian'
        save_file = fullfile(dirs.results,sprintf('Fits_NoCB_Expt3_Gen1.mat'));
end

save(save_file,'Fit');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Run model with best-fit parameters                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch Fit.Model
    case 'ConfirmationBias'
        save_file = fullfile(dirs.results,sprintf('Fits_CB_Expt3_Gen1.mat'));
    case 'Bayesian'
        save_file = fullfile(dirs.results,sprintf('Fits_NoCB_Expt3_Gen1.mat'));
end

load(save_file)

for s = 1:nSub
    
    thisData = dSocial(dSocial(:,1) == Sub(s),:);
    
    fprintf('Subject %i \n',s);
    
    % Get Advisor Correct
    for j = 1:3
        AdvisorCorrect(:,j) = thisData(thisData(:,5) == j,6);
        Choice(:,j) = thisData(thisData(:,5) == j,7);
    end
    
    choice = [Choice(:,1); Choice(:,2); Choice(:,3)];
    
    Fit.NTrials(s) = length(thisData);
    
    x(1) = Fit.Result.BestFit(s,2);
    x(2) = Fit.Result.BestFit(s,3);
    x(3) = Fit.Result.BestFit(s,4);
    
    Fit.Priors.Use(:) = 0;
    
    [lik(s,1) latents{s,1}] = jointfit_model(AdvisorCorrect,choice, Fit.Priors,x,Fit.Model);
    
    AIC(s,1) = 2*lik(s) + (2*Fit.Nparms*(Fit.Nparms+1))/(Fit.NTrials(s)-Fit.Nparms-1);
    BIC(s,1) = 2*lik(s) + Fit.Nparms * log(Fit.NTrials(s));
end

g_mean(:,1) = exp(-lik ./ Fit.NTrials');
gm_AIC(:,1) = exp(-0.5 .* AIC ./ Fit.NTrials');
gm_BIC(:,1) = exp(-0.5 .* BIC ./ Fit.NTrials');

switch Fit.Model
    case 'ConfirmationBias'
        save_file = fullfile(dirs.results,sprintf('Fits_CB_Expt3_Gen1.mat'));
    case 'Bayesian'
        save_file = fullfile(dirs.results,sprintf('Fits_NoCB_Expt3_Gen1.mat'));
end

fprintf('log lik = %0.1f(%0.1f), AIC = %0.1f(%0.1f), avg lik = %0.2f \n',...
    mean(lik),std(lik)/sqrt(nSub),mean(AIC),std(AIC)/sqrt(nSub),mean(gm_AIC)); 

save(save_file,'Fit','lik','latents','g_mean','gm_AIC','gm_BIC','AIC','BIC');
    

 


