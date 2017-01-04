%% sim_model.m
% This code simulates the model performing the task for 1000 iterations
% For each iteration, the model takes the optimal priors, and the optimal best-fit beta for each sub  on 36 trials of a 75% advisor, 36 trials of a 
% 50% advisor and 36 trials with a 25% advisor. 
% Saves results of this simulation in a 3 (advisors) x 36 (trials) x 200 (iterations)  Matrix 

clear mex
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%                                              Set Directories & Load data                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dirs.data = '../../data';
dirs.results = 'interm_results';

load(fullfile(dirs.data,'AllData.mat'));
%load(fullfile(dirs.results,'Fits_Advisor_Latents.mat'));

addpath('../models');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%                                                 Set Script Parameters                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% Subjects
Sub = [101 102 103 104 105 106 107 108 109 110 112 113 114 115 116 118 119 120 121 122 123 124 125 126 127 128];
nSub = length(Sub);

% Colors
col_code(1,:) = [0.2980392156862745, 0.4470588235294118, 0.6901960784313725];
col_code(2,:) = [0.3333333333333333, 0.6588235294117647, 0.40784313725490196];
col_code(3,:) = [0.7686274509803922, 0.3058823529411765, 0.3215686274509804];
err_col(1,:) = [0.4, 0.6, 0.9];
err_col(2,:) = [0.4, 0.8, 0.5];
err_col(3,:) = [0.9, 0.4, 0.4];

% Font Size
font_size = 16;

% Number of Iterations
n_iterations = 500;

% Number of trials per advisor
n_trials = 36;

% Refit the model?
refit_model = 1;

% Model
model = 'CB_Learner'; 

% Priors
switch model
    case 'CB_Learner'
        prior = [2.6,1.6];
    case 'NoCB_Learner'
        prior = [2.1,1.3];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
%                                                   Refit Model                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% Refit the model to obtain softmax temperature for each subject
for i = 1
if refit_model 
    
    Fit.Subjects = Sub;
    switch model
        case 'CB_Learner'
            Fit.Model = 'NewCB_Learner';
        case 'NoCB_Learner'
            Fit.Model = 'NoCB_Learner';
    end
    Fit.NIter = 3; 
    Fit.Start = ones(1,length(Fit.Subjects));
    Fit.End = ones(1,length(Fit.Subjects))*108;
    
    Fit.Nparms = 1;
    Fit.LB = 1e-6*ones(1,Fit.Nparms);
    Fit.UB = [inf];
    
    Fit.Priors.Use(1) = 1;   % use (gamma) priors on the Beta (softmax) parameter?
    Fit.Priors.Parms(1,1) = 2;
    Fit.Priors.Parms(1,2) = 3;
        
    Fit.Priors.Use(2) = 1;   % use (gamma) priors on the Beta (softmax) parameter?
    Fit.Priors.Parms(2,1) = 8;
    
    x = [0.01:0.01:0.99];
    y1 = betapdf(x,prior(1),prior(2));
    betaprior = y1;
    alpha = 1;
    
    for s = 1:nSub
        fprintf('Subject %d... (index %d) \n',Fit.Subjects(s),s)
        thisData = AllData{s,3}.Learn{1,1};
        for j = 1:3
            AdvisorCorrect(:,j) = thisData.AdvisorCorrect(find(thisData.Advisor == j));
        end
        
        % Which model to run
        switch model
            case 'CB_Learner'
                A1 = NewCB_Learner(AdvisorCorrect(:,1),betaprior',alpha);
                A2 = NewCB_Learner(AdvisorCorrect(:,2),betaprior',alpha);
                A3 = NewCB_Learner(AdvisorCorrect(:,3),betaprior',alpha);
            case 'NoCB_Learner'
                A1 = NoCB_Learner(AdvisorCorrect(:,1),betaprior',alpha);
                A2 = NoCB_Learner(AdvisorCorrect(:,2),betaprior',alpha);
                A3 = NoCB_Learner(AdvisorCorrect(:,3),betaprior',alpha);
        end
                
        fit_p{s,1} = AllData{s,1};
        fit_p{s,2}.p_dist = [A1.p_dist(1:end-1,:); A2.p_dist(1:end-1,:); A3.p_dist(1:end-1,:)];
        fit_p{s,2}.pUP = [A1.pUP(1:end-1); A2.pUP(1:end-1); A3.pUP(1:end-1)];
        fit_p{s,2}.betaprior = betaprior;
        fit_p{s,2}.betaparms = prior;
        
        
        %% Fit pUP to choice
        thisData = AllData{s,3}.Learn{1,1};
        Fit.NTrials(s) = sum(thisData.ValidTrials);
        nTrials = length(thisData.ValidTrials);
        pHat = [fit_p{s,2}.pUP(1:end)];
        pHat(find(isnan(pHat))) = pHat(find(isnan(pHat))-1);
        
        for j = 1:3
            Choice(:,j) = thisData.Choice(find(thisData.Advisor == j));
        end
        
        choice = [Choice(:,1); Choice(:,2); Choice(:,3)];
        
        for iter = 1:Fit.NIter
            Fit.init(s,iter,[1]) = rand*5;
            
            [res,lik,flag,out,lambda,grad,hess] = ...
                fmincon(@(x) basic_bayes(pHat,choice,Fit.Priors,x,prior(1),prior(2)),...
                Fit.init(s,iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
                'GradObj','off','DerivativeCheck','off','LargeScale','off','Algorithm','active-set','Hessian','off','Display','off'));
            
            Fit.Result.Beta(s,:,iter) = res(1);
            Fit.Result.Hessian(s,:,:,iter) = full(hess); % save the Hessian to do Laplace approx later; it was sparse initially, so use "full" to expand
            Fit.Result.Lik(s,iter) = lik;
            
            %Calculate BIC here...
            Fit.Result.BIC(s,iter) = lik + (Fit.Nparms/2*log(Fit.NTrials(s)));
            Fit.Result.BIC(s,iter) = lik + (Fit.Nparms/2*log(Fit.NTrials(s)));
            Fit.Result.AverageBIC(s,iter) = -Fit.Result.BIC(s,iter)/Fit.NTrials(s);
            Fit.Result.CorrectedLikPerTrial(s,iter) = exp(Fit.Result.AverageBIC(s,iter));
            
            [[1:s]' Fit.Result.CorrectedLikPerTrial]  % to view progress so far
        end
    end
    
    %Saving Data
    [a,b] = min(Fit.Result.Lik,[],2);
    d = length(hess); % how many parameters are we fitting
    
    for s = 1:length(Fit.Subjects)
        Fit.Result.BestFit(s,:) = [Fit.Subjects(s),...
            Fit.Result.Beta(s,b(s)),...
            Fit.Result.Lik(s,b(s)),...
            Fit.Result.BIC(s,b(s)),...
            Fit.Result.AverageBIC(s,b(s)),...
            Fit.Result.CorrectedLikPerTrial(s,b(s))];
            % compute Laplace approximation at the ML point, using the Hessian
        
        Fit.Result.Laplace(s) = -a(s) + 0.5*d*log(2*pi) - 0.5*log(det(squeeze(Fit.Result.Hessian(s,:,:,b(s)))));
    end
    Fit.Result.BestFit
    
    switch model
        case 'CB_Learner'
            save_file = fullfile(dirs.results,sprintf('Fits_Advisor_optParms_NewCB'));
        case 'NoCB_Learner'
            save_file = fullfile(dirs.results,sprintf('Fits_Advisor_optParms_NoCB'));
    end
    
    
    save (save_file,'Fit','fit_p')
    
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
%                                                 Run Simulation                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% Initialize Output
sim_model_res = NaN(3,n_trials,n_iterations);

% Extract latents
switch model
    case 'CB_Learner'
        load(fullfile(dirs.results,sprintf('Fits_Advisor_optParms_NewCB')));
    case 'NoCB_Learner'
        load(fullfile(dirs.results,sprintf('Fits_Advisor_optParms_NoCB')));
end

% Model Parameters
x = [0.01:0.01:0.99];
y1 = betapdf(x,prior(1),prior(2));
betaprior = y1;
alpha = 1;

% run iterations
for it = 1:n_iterations
    
    fprintf('Iteration %i \n',it);
    
    % results from this iteration
    sim_1_it = NaN(nSub, 3, n_trials);
    
    for s = 1:nSub
         
        % Go through each advisor
        for a = 1:3
            
            % Generate outcomes for each advisor
            switch a
                case 1
                    AdvisorCorrect = [ones(0.75*n_trials,1);zeros(0.25*n_trials,1)];
                case 2
                    AdvisorCorrect = [ones(0.50*n_trials,1);zeros(0.50*n_trials,1)];
                case 3
                    AdvisorCorrect = [ones(0.25*n_trials,1);zeros(0.75*n_trials,1)];
                    
            end
            
            % Shuffle Advisor Correct
            AdvisorCorrect = AdvisorCorrect(randperm(length(AdvisorCorrect)));
            
            % Compute pHat for this advisor
            switch model
                case 'CB_Learner'
                     A1 = NewCB_Learner(AdvisorCorrect(1:n_trials-1,1),betaprior',alpha);
                case 'NoCB_Learner'
                     A1 = NoCB_Learner(AdvisorCorrect(1:n_trials-1,1),betaprior',alpha);
            end

            % This subject's Beta
            thisBeta = Fit.Result.BestFit(s,2);
            
            % Predict choice probabilty for this subject
            pFor =  1./(1+exp(-thisBeta*(A1.pUP-0.5)));
            
            % Simulate actual choices by this subject
            rand_no = rand(n_trials,1);
            
            % Did the subject make the choice
            sim_1_it(s,a,:) = rand_no < pFor;

        end

    end
    
    % Save average results for this iteration
    sim_model_res(1,:,it) = mean(sim_1_it(:,1,:),1);
    sim_model_res(2,:,it) = mean(sim_1_it(:,2,:),1);
    sim_model_res(3,:,it) = mean(sim_1_it(:,3,:),1);

end

save_file = fullfile(dirs.results,sprintf('sim_%s_%i',model,n_iterations));
save (save_file,'sim_model_res')


