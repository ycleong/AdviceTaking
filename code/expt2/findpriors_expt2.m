%% Find Betas
% This script performs the model-fitting for the Advisor Evaluation Phase for Expt 2
% Assumes a different prior for each advisor
% The best-fit priors for the Confirmation Bias Model
%   1-star advisor: h1 = 0.2; t1 = 0.8
%   2-star advisor: h2 = 1.5; t2 = 3.1;
%   3-star advisor: h3 = 3.0; t3 = 0.9;
%   4-star advisor: h4 = 1.6; t4 = 0.1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                             Set Directories Paths and Script Parameters                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear mex
clear all

col_code(1,:) = [0.2980392156862745, 0.4470588235294118, 0.6901960784313725];
col_code(2,:) = [0.3333333333333333, 0.6588235294117647, 0.40784313725490196];
col_code(3,:) = [0.7686274509803922, 0.3058823529411765, 0.3215686274509804];

% Set Directories
dirs.data = '../../data';
dirs.results = 'interm_results';
load(fullfile(dirs.data,'AllData_Expt2.mat'));
addpath('../models');

Sub = [101:130];
nSub = length(Sub);

% Which model to fit
Model = 'Prob_Learner'; % Options: CB_Learner and Prob_Learner

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    Fitting Parameters                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear Fit
clear fit_p

min_sumLogLik = 3000;
opt_parm = [0,0,0,0,0,0,0,0,0];

Fit.Subjects = Sub;
Fit.Model = Model;
Fit.NIter = 3; 
Fit.Start = ones(1,length(Fit.Subjects));
Fit.End = ones(1,length(Fit.Subjects))*108;

Fit.Nparms = 1;
Fit.LB = 0;
Fit.UB = [inf];

Fit.Priors.Use(1) = 1;   % use (gamma) priors on the Beta (softmax) parameter?
Fit.Priors.Parms(1,1) = 2;
Fit.Priors.Parms(1,2) = 3;

Fit.Priors.Use(2) = 1;   % use (gamma) priors on the Beta (softmax) parameter?
Fit.Priors.Parms(2,1) = 8;

op = 1;
all_opt_parms = NaN(1,9);

for t1 = 1:5
for t2 = 1:5
for t3 = 1:5
for t4 = 1:5

for h1 = 1:5
for h2 = 1:5
for h3 = 1:5
for h4 = 1:5
    alpha = 1;
    
    fprintf('H1 = %i, T1 = %i, H2 = %i, T2 = %i, H3 = %i, T3 = %i, H4 = %i, T4 = %i \n',h1,t1,h2,t2,h3,t3,h4,t4);
    x = [0.01:0.01:0.99];
    y1 = betapdf(x,h1,t1);
    y2 = betapdf(x,h2,t2);
    y3 = betapdf(x,h3,t3);
    y4 = betapdf(x,h4,t4);
    
    for s = 1:nSub
        thisData = AllData{s,3}.Learn{1,1};
        for j = 1:4
            AdvisorCorrect(:,j) = thisData.AdvisorCorrect(find(thisData.Advisor == j));
        end
        
        switch Model
            case 'CB_Learner'
                A1 = NewCB_Learner(AdvisorCorrect(:,1),y1',alpha);
                A2 = NewCB_Learner(AdvisorCorrect(:,2),y2',alpha);
                A3 = NewCB_Learner(AdvisorCorrect(:,3),y3',alpha);
                A4 = NewCB_Learner(AdvisorCorrect(:,4),y4',alpha);
            case 'Prob_Learner'
                A1 = NoCB_Learner(AdvisorCorrect(:,1),y1',alpha);
                A2 = NoCB_Learner(AdvisorCorrect(:,2),y2',alpha);
                A3 = NoCB_Learner(AdvisorCorrect(:,3),y3',alpha);
                A4 = NoCB_Learner(AdvisorCorrect(:,4),y4',alpha);
        end
        
        fit_p{s,1}.p_dist = [A1.p_dist(1:end-1,:); A2.p_dist(1:end-1,:); A3.p_dist(1:end-1,:); A4.p_dist(1:end-1,:)];
        fit_p{s,1}.pUP = [A1.pUP(1:end-1); A2.pUP(1:end-1); A3.pUP(1:end-1); A4.pUP(1:end-1)];
    end
    
    %% Fit pUP to choice
    for s = 1:nSub
        thisData = AllData{s,3}.Learn{1,1};
        Fit.NTrials(s) = length(A1.pUP) - sum(isnan(A1.pUP) - 1);
        nTrials = length(A1.pUP) - 1;
        pHat = [fit_p{s,1}.pUP(1:end)];
        pHat(find(isnan(pHat))) = pHat(find(isnan(pHat))-1);
        
        for j = 1:4
            Choice(:,j) = thisData.Choice(find(thisData.Advisor == j));
        end
        
        choice = [Choice(:,1); Choice(:,2); Choice(:,3); Choice(:,4)];
        
        for iter = 1:Fit.NIter
            Fit.init(s,iter,[1]) = rand*5;
            
            [res,lik,flag,out,lambda,grad,hess] = ...
                fmincon(@(x) basic_bayes4(pHat,choice,Fit.Priors,x,h1,t1,h2,t2,h3,t3,h4,t4),...
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
        Beta(s) = temp.Beta(s,b(s));
    end
    LogLik;
    sumLogLik = sum(LogLik);
    
    if sumLogLik < min_sumLogLik
        min_sumLogLik = sumLogLik;
        opt_parm = [alpha,h1,t1,h2,t2,h3,t3,h4,t4];
        
        for s = 1:length(Fit.Subjects)
            Beta(s) = temp.Beta(s,b(s));
        end
        all_opt_parms(op,:) = opt_parm;
        op = op + 1;
    end
    
    fprintf('This log lik = %0.3f, Min log lik = %0.3f, OptParm = [%0.2f %i %i %i %i %i %i %i %i] \n',sumLogLik, min_sumLogLik,opt_parm);
end
end
end
end
end
end
end
end

% Save data
switch Model
    case 'CB_Learner'
        save_file = fullfile(dirs.results,sprintf('Expt2_BestParms_CB.mat'));
    case 'Prob_Learner'
        save_file = fullfile(dirs.results,sprintf('Expt2_BestParms_NoCB.mat'));
end

save(save_file,'opt_parm','min_sumLogLik','all_opt_parms','Beta');
    