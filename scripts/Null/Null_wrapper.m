%% Null Model Wrapper
% This model fits a constant p(bet) for each participant

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                  Setup                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear mex
clear all

% Set Directories and load data
dirs.data = '../../data';
dirs.results = '../../results';
addpath('../../scripts');

% Which experiment to run ('Expt1' or Expt2')
experiment = 'Expt2';

switch experiment 
    case 'Expt1'
        load(fullfile(dirs.data,'AllData.mat'));
        n_advisor = 3;
    case 'Expt2'
        load(fullfile(dirs.data,'AllData_Expt2.mat'));
        n_advisor = 4;
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Initialize parameters for model-fitting                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subjects
switch experiment
    case 'Expt1'
        Sub = [101 102 103 104 105 106 107 108 109 110 112 113 114 115 116 118 119 120 121 122 123 124 125 126 127 128];
    case 'Expt2'
        Sub = [101:130];      
end

nSub = length(Sub);

% Set fitting parameters
Fit.Subjects = Sub;
Fit.Model = 'Null';
Fit.NIter = 3; % how many iterations of fits to run

Fit.Start = ones(1,length(Fit.Subjects));
Fit.End = ones(1,length(Fit.Subjects))*108;

Fit.Nparms = 1; % pSW and pSL
Fit.LB = [1e-6];
Fit.UB = [inf];

Fit.Priors.Use(1) = 0;   % use (gamma) priors on the Beta (softmax) parameter?
Fit.Priors.Parms(1,1) = 0;
Fit.Priors.Parms(1,2) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Fit Model                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for s = 1:nSub
    thisData = AllData{s,3}.Learn{1,1};
    
    fprintf('Subject %i \n',Sub(s));
    
    % Get Advisor Correct and Choice
    for j = 1:n_advisor
        AdvisorCorrect(:,j) = thisData.AdvisorCorrect(find(thisData.Advisor == j));
        Choice(:,j) = thisData.Choice(find(thisData.Advisor == j));
    end
    
    Fit.NTrials(s) = sum(thisData.ValidTrials);
    nTrials = length(thisData.ValidTrials);

    % Fit model 3 times
    for iter = 1:Fit.NIter
       Fit.init(s,iter,[1]) = 5*rand;
       
       [res,lik,flag,out,lambda,grad,hess] = ...
           fmincon(@(x) Null_lik(AdvisorCorrect,Choice, Fit.Priors,x,Fit.Model,n_advisor),...
           Fit.init(s,iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
           'GradObj','off','DerivativeCheck','off','LargeScale','off','Algorithm','active-set','Hessian','off','Display','on'));
        
        Fit.Result.Beta(s,:,iter) = res(1);
        Fit.Result.Hessian(s,:,:,iter) = full(hess); % save the Hessian to do Laplace approx later; it was sparse initially, so use "full" to expand
        Fit.Result.Lik(s,iter) = lik;
        
    end
        
end

% Find best fit parameters                
[a,b] = min(Fit.Result.Lik,[],2);

for s = 1:length(Fit.Subjects)
    Fit.Result.BestFit(s,:) = [Fit.Subjects(s),...
        Fit.Result.Beta(s,b(s)),...
        Fit.Result.Lik(s,b(s))];
end


for s = 1:nSub
    
    thisData = AllData{s,3}.Learn{1,1};
    
    % Get Advisor Correct and Choice
    for j = 1:n_advisor
        AdvisorCorrect(:,j) = thisData.AdvisorCorrect(find(thisData.Advisor == j));
        Choice(:,j) = thisData.Choice(find(thisData.Advisor == j));
    end
    
    nTrials = length(thisData.ValidTrials);
    
    x(1) = Fit.Result.BestFit(s,2);
    
    [lik(s,1) latents{s,1}] = Null_lik(AdvisorCorrect,Choice, Fit.Priors,x,Fit.Model,n_advisor);
    
    AIC(s,1) = 2*lik(s) + Fit.Nparms*2 + (2*Fit.Nparms*(Fit.Nparms+1))/(Fit.NTrials(s)-Fit.Nparms-1);
    BIC(s,1) = 2*lik(s) + Fit.Nparms * log(Fit.NTrials(s));
    
end

g_mean(:,1) = exp(-lik ./ Fit.NTrials');
gm_AIC(:,1) = exp(-0.5 .* AIC ./ Fit.NTrials');
gm_BIC(:,1) = exp(-0.5 .* BIC ./ Fit.NTrials');

fprintf('log lik = %0.1f(%0.1f), AIC = %0.1f(%0.1f), avg lik = %0.2f \n',...
    mean(lik),std(lik)/sqrt(nSub),mean(AIC),std(AIC)/sqrt(nSub),mean(gm_AIC));

%% Save data
switch experiment
    case 'Expt1'
        save_file = fullfile(dirs.results,sprintf('Fits_Null_Expt1.mat'));
    case 'Expt2'
        save_file = fullfile(dirs.results,sprintf('Fits_Null_Expt2.mat'));   
end
save (save_file,'Fit','lik','latents','g_mean','gm_AIC','gm_BIC','AIC','BIC');



