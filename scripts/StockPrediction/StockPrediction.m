%% Modeling Stock Prediction phase
%  run_BO_stock: Fit Bayesian learning model to stock prediciton phase data
%  run_fit_self: Fit Bayesian learning model to participants' choices (now obsolete)

clear mex
clear all

%% Set Directories & Load data & Set Script functionality
dirs.data = '../../data';
dirs.results = '../../results';
load(fullfile(dirs.data,'AllData.mat'));
AllData_Expt2 = load(fullfile(dirs.data,('AllData_Expt2.mat')));
addpath(genpath('../../scripts'));

run_BO_stock = 1;
run_fit_self = 0;

%% Subjects
Sub = [101 102 103 104 105 106 107 108 109 110 112 113 114 115 116 118 119 120 121 122 123 124 125 126 127 128];
Sub2 = [101:130];
nSub = length(Sub);
nSub2 = length(Sub2);

% Run BO Stock
if run_BO_stock 
    for z = 1
    fprintf('Running BO model on Self Phase: Study 1 \n');
    %% Run BO Stock       
       for i = 1:nSub
           fprintf('Running index %i of %i \n',i,nSub);
           thisOutcome = AllData{i,3}.NoAdvice{1,1}.StockOutcome;
           fit_p{i,1} = AllData{i,1};
           
           [fit_p{i,2}] = ProbabilityLearner(thisOutcome);
           fit_p{i,2}.name = 'ProbLearner_NoAdvice';
       end 
       output_file = fullfile(dirs.results,'BayesianStockPs.mat');
       save(output_file, 'fit_p');
       disp('Ran BO Stock Study1')
       
       clear fit_p;
       
       %% Run BO Stock
       for i = 1:nSub2
           fprintf('Running index %i of %i \n',i,nSub2);
           thisOutcome = AllData_Expt2.AllData{i,3}.NoAdvice{1,1}.StockOutcome;
           fit_p{i,1} = AllData_Expt2.AllData{i,1};
           
           [fit_p{i,2}] = ProbabilityLearner(thisOutcome);
           fit_p{i,2}.name = 'ProbLearner_NoAdvice';
       end
       output_file = fullfile(dirs.results,'BayesianStockPs_Study2.mat');
       save(output_file, 'fit_p');
       disp('Ran BO Stock Study2')
       
    end
end

% Run Fit (Self)
if run_fit_self
    for z = 1    
    fprintf('Fit Self Phase choices  \n');
    load(fullfile(dirs.results,'BayesianStockPs.mat'));    
   
    Fit.Subjects = Sub;
    Fit.Model = 'Problearner';
    Fit.NIter = 3; % how many iterations of fits to run
    
    Fit.Start = ones(1,length(Fit.Subjects)); 
    Fit.End = ones(1,length(Fit.Subjects))*100;
    
    Fit.Nparms = 1;
    Fit.LB = 1e-6*ones(1,Fit.Nparms);
    Fit.UB = [inf];
    
    Fit.Priors.Use(1) = 0;   % use (gamma) priors on the Beta (softmax) parameter?
    Fit.Priors.Parms(1,1) = 2;
    Fit.Priors.Parms(1,2) = 3;
    
    Fit.Priors.Use(2) = 0;   % use (gamma) priors on the Beta (softmax) parameter?
    Fit.Priors.Parms(1,1) = 2;
    Fit.Priors.Parms(1,2) = 3;
    
    for s = 1:nSub
        fprintf('Subject %d... (index %d) \n',Fit.Subjects(s),s)
        thisData = AllData{s,3}.NoAdvice{1,1};
        Fit.NTrials(s) = sum(thisData.ValidTrials);
        nTrials = length(thisData.ValidTrials);
        pHat = [fit_p{s,2}.pUP(1:end-1)];
        pHat(find(isnan(pHat))) = pHat(find(isnan(pHat))-1);
        choice = thisData.Choice;
            
        for iter = 1:Fit.NIter
            Fit.init(s,iter,[1]) = rand*5;

            [res,lik,flag,out,lambda,grad,hess] = ...
                fmincon(@(x) basic_bayes(pHat,choice,Fit.Priors,x),...
                Fit.init(s,iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
                'GradObj','off','DerivativeCheck','off','LargeScale','off','Algorithm','active-set','Hessian','off'));

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
    
    save_file = fullfile(dirs.results,sprintf('Fits_Self_optParms'));
    save (save_file,'Fit')  
    
    %Get Latents
    for s = 1:nSub
        Fit.Priors.Use(1) = 0;
        
        thisData = AllData{s,3}.NoAdvice{1,1};
        pHat = [fit_p{s,2}.pUP(1:end-1)];
        pHat(find(isnan(pHat))) = pHat(find(isnan(pHat))-1);
        choice = thisData.Choice;
        Beta_hat = Fit.Result.BestFit(s,2);
        [lik,latents{s,1}] = basic_bayes(pHat,choice,Fit.Priors,Beta_hat);
    end
    
    save_file = fullfile(dirs.results,sprintf('Fits_Self_Latents'));
    save (save_file,'latents')
        
   end    
end

