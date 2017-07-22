%% RL Wrapper
% Fits the data to a reinforcement learning model with a temporal difference learning rule 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              Boiler plate                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear mex
clear all

col_code(1,:) = [0.2980392156862745, 0.4470588235294118, 0.6901960784313725];
col_code(2,:) = [0.3333333333333333, 0.6588235294117647, 0.40784313725490196];
col_code(3,:) = [0.7686274509803922, 0.3058823529411765, 0.3215686274509804];

% Set Directories and load data
dirs.data = '../../data';
dirs.results = '../../results';
addpath('../../scripts');

% Which generation to fit? 'gen1' or 'gen2'
generation = 'gen2';

switch generation 
    case 'gen1'
        dSocial = csvread(fullfile(dirs.data,'dSocial_gen1.csv'),1,1);
        n_advisor = 3;
    case 'gen2'
        dSocial = csvread(fullfile(dirs.data,'dSocial_gen2.csv'),1,1);
        n_advisor = 3;
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Initialize parameters for model-fitting                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subjects
Sub = unique(dSocial(:,1));
nSub = length(Sub);

% Set fitting parameters
Fit.Subjects = Sub;
Fit.Model = 'RL';
Fit.NIter = 3; % how many iterations of fits to run

Fit.Start = ones(1,length(Fit.Subjects));
Fit.End = ones(1,length(Fit.Subjects))*108;

Fit.Nparms = 2; % Eta Beta
Fit.LB = [1e-6 1e-6];
Fit.UB = [1 Inf];

Fit.Priors.Use(1) = 0;   % use (gamma) priors on the Beta (softmax) parameter?
Fit.Priors.Parms(1,1) = 0;
Fit.Priors.Parms(1,2) = 0;

Fit.Priors.Use(2) = 1;   % use (gamma) priors on the Beta (softmax) parameter?
Fit.Priors.Parms(2,1) = 2;
Fit.Priors.Parms(2,2) = 3;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Fit Model                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for s = 1:nSub
    thisData = dSocial(dSocial(:,1) == Sub(s),:);

    fprintf('Subject %i \n',Sub(s));
    
    % Get Advisor Correct and Choice
    for j = 1:n_advisor
        AdvisorCorrect(:,j) = thisData(thisData(:,5) == j,6);
        Choice(:,j) = thisData(thisData(:,5) == j,7);
    end
    
    Fit.NTrials(s) = length(thisData);

    % Fit model 3 times
    for iter = 1:Fit.NIter
       Fit.init(s,iter,[1]) = rand;
       Fit.init(s,iter,[2]) = rand*5;
       
       [res,lik,flag,out,lambda,grad,hess] = ...
           fmincon(@(x) RL_lik(AdvisorCorrect,Choice, Fit.Priors,x,Fit.Model,n_advisor),...
           Fit.init(s,iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
           'GradObj','off','DerivativeCheck','off','LargeScale','off','Algorithm','active-set','Hessian','off','Display','on'));
        
        Fit.Result.Eta(s,:,iter) = res(1);
        Fit.Result.Beta(s,:,iter) = res(2);
        Fit.Result.Hessian(s,:,:,iter) = full(hess); % save the Hessian to do Laplace approx later; it was sparse initially, so use "full" to expand
        Fit.Result.Lik(s,iter) = lik;
    end
        
end

% Find best fit parameters                
[a,b] = min(Fit.Result.Lik,[],2);

for s = 1:length(Fit.Subjects)
    Fit.Result.BestFit(s,:) = [Fit.Subjects(s),...
        Fit.Result.Eta(s,b(s)),...
        Fit.Result.Beta(s,b(s)),...
        Fit.Result.Lik(s,b(s))];
end

for s = 1:nSub
    
    thisData = dSocial(dSocial(:,1) == Sub(s),:);

    fprintf('Subject %i \n',Sub(s));
    
    % Get Advisor Correct and Choice
    for j = 1:n_advisor
        AdvisorCorrect(:,j) = thisData(thisData(:,5) == j,6);
        Choice(:,j) = thisData(thisData(:,5) == j,7);
    end
    
    Fit.NTrials(s) = length(thisData);
    
    x(1) = Fit.Result.BestFit(s,2);
    x(2) = Fit.Result.BestFit(s,3);
    
    [lik(s,1) latents{s,1}] = RL_lik(AdvisorCorrect,Choice, Fit.Priors,x,Fit.Model,n_advisor);
    
    AIC(s,1) = 2*lik(s) + Fit.Nparms*2 + (2*Fit.Nparms*(Fit.Nparms+1))/(Fit.NTrials(s)-Fit.Nparms-1);
    BIC(s,1) = 2*lik(s) + Fit.Nparms * log(Fit.NTrials(s));
end

g_mean(:,1) = exp(-lik ./ Fit.NTrials');
gm_AIC(:,1) = exp(-0.5 .* AIC ./ Fit.NTrials');
gm_BIC(:,1) = exp(-0.5 .* BIC ./ Fit.NTrials');

fprintf('log lik = %0.1f(%0.1f), AIC = %0.1f(%0.1f), avg lik = %0.2f \n',...
    mean(lik),std(lik)/sqrt(nSub),mean(AIC),std(AIC)/sqrt(nSub),mean(gm_AIC));


%% Save data
%% Save data
switch generation
    case 'gen1'
        save_file = fullfile(dirs.results,sprintf('Fits_RL_Expt3_Gen1.mat'));
    case 'gen2'
        save_file = fullfile(dirs.results,sprintf('Fits_RL_Expt3_Gen2.mat'));   
end
save(save_file,'Fit','lik','latents','g_mean','gm_AIC','gm_BIC','AIC','BIC');
