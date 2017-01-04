%% Findpriors.m
% Finds the MAP estimate of the values for alpha and beta. 
% Assumes same prior for all advisors
% Note: For speed purposes here, we search in increments of 1.
%       In the actual experiment, we search in increments of 0.1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              Boilet plate                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear mex
clear all

col_code(1,:) = [0.2980392156862745, 0.4470588235294118, 0.6901960784313725];
col_code(2,:) = [0.3333333333333333, 0.6588235294117647, 0.40784313725490196];
col_code(3,:) = [0.7686274509803922, 0.3058823529411765, 0.3215686274509804];

% Set Directories and load data
dirs.data = '../../data';
dirs.results = 'interm_results';
load(fullfile(dirs.data,'AllData.mat'));
addpath('../models');

% Subjects
Sub = [101 102 103 104 105 106 107 108 109 110 112 113 114 115 116 118 119 120 121 122 123 124 125 126 127 128];
nSub = length(Sub);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Initialize parameters for model-fitting                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize variables
min_sumLogLik = 2000;
opt_parm = [0,0];

% Set fitting parameters
Fit.Subjects = Sub;
Fit.Model = 'Problearner';
Fit.NIter = 3; % how many iterations of fits to run

Fit.Start = ones(1,length(Fit.Subjects));
Fit.End = ones(1,length(Fit.Subjects))*108;

Fit.Nparms = 1;
Fit.LB = 1e-6*ones(1,Fit.Nparms);
Fit.UB = [inf];

Fit.Priors.Use(1) = 1;   % use (gamma) priors on the Beta (softmax) parameter?
Fit.Priors.Parms(1,1) = 2;
Fit.Priors.Parms(1,2) = 3;

Fit.Priors.Use(2) = 0;   % exponential prior on alpha and beta
Fit.Priors.Parms(2,1) = 8;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                               Fit Model                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
op = 1;
all_opt_parms = NaN(1,2);

for h1 = 1:5
    for t1 = 1:5
                fprintf('H1 = %i, T1 = %i \n',h1,t1);
                
                x = [0.01:0.01:0.99];
                y1 = betapdf(x,h1,t1);
                betaprior = y1;
                alpha = 1;

                % Fit confirmation bias model to observed outcomes of advisors               
                for s = 1:nSub
                    thisData = AllData{s,3}.Learn{1,1};
                    for j = 1:3
                        AdvisorCorrect(:,j) = thisData.AdvisorCorrect(find(thisData.Advisor == j));
                    end
                    
                    A1 = NewCB_Learner(AdvisorCorrect(:,1),betaprior',alpha);
                    A2 = NewCB_Learner(AdvisorCorrect(:,2),betaprior',alpha);
                    A3 = NewCB_Learner(AdvisorCorrect(:,3),betaprior',alpha);
                    
                    fit_p{s,1}.p_dist = [A1.p_dist(1:end-1,:); A2.p_dist(1:end-1,:); A3.p_dist(1:end-1,:)];
                    fit_p{s,1}.pUP = [A1.pUP(1:end-1); A2.pUP(1:end-1); A3.pUP(1:end-1)];
                    fit_p{s,1}.betaprior = betaprior;
                    fit_p{s,1}.betaparms = [h1 t1];
                end
                
                % Fit p(a) to choices    
                for s = 1:nSub
                    thisData = AllData{s,3}.Learn{1,1};
                    Fit.NTrials(s) = sum(thisData.ValidTrials);
                    nTrials = length(thisData.ValidTrials);
                    pHat = [fit_p{s,1}.pUP(1:end)];
                    pHat(find(isnan(pHat))) = pHat(find(isnan(pHat))-1);
                    
                    for j = 1:3
                        Choice(:,j) = thisData.Choice(find(thisData.Advisor == j));
                    end
                    
                    choice = [Choice(:,1); Choice(:,2); Choice(:,3)];
                    
                    for iter = 1:Fit.NIter
                        Fit.init(s,iter,[1]) = rand*5;
                        
                        [res,lik,flag,out,lambda,grad,hess] = ...
                            fmincon(@(x) basic_bayes(pHat,choice,Fit.Priors,x,h1,t1),...
                            Fit.init(s,iter,:),[],[],[],[],Fit.LB,Fit.UB,[],optimset('maxfunevals',5000,'maxiter',2000,...
                            'GradObj','off','DerivativeCheck','off','LargeScale','off','Algorithm','active-set','Hessian','off','Display','off'));
                        
                        temp.Beta(s,:,iter) = res(1);
                        temp.Hessian(s,:,:,iter) = full(hess); % save the Hessian to do Laplace approx later; it was sparse initially, so use "full" to expand
                        temp.Lik(s,iter) = lik;
                    end
                end
                
                [a,b] = min(temp.Lik,[],2);
                for s = 1:length(Fit.Subjects)
                    LogLik(s) = temp.Lik(s,b(s));
                end
                LogLik;
                sumLogLik = sum(LogLik);
                % Add regularizing priors
                sumLogLik = sumLogLik - log(exppdf(h1,Fit.Priors.Parms(2,1)));
                sumLogLik = sumLogLik - log(exppdf(t1,Fit.Priors.Parms(2,1)));
                
                if sumLogLik < min_sumLogLik
                    min_sumLogLik = sumLogLik;
                    opt_parm = [h1,t1];
                    
                    all_opt_parms(op,:) = opt_parm;
                    op = op + 1;
                end
                
                fprintf('This log lik = %0.3f, Min log lik = %0.3f, OptParm = [%i %i] \n',sumLogLik, min_sumLogLik,opt_parm);
    end
end
    
%% Save data
save_file = fullfile(dirs.results,sprintf('BestParms.mat'));
save (save_file,'opt_parm','min_sumLogLik');

opt_parm
min_sumLogLik
    



