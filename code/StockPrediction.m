%% Generate Figure: Study 1 Self Phase
% Panel A: Simulated data stock
% Panel B: Number of trials predicted
% Panel C: Number of trials pred

clear mex
clear all

%% Set Directories & Load data & Set Script functionality
dirs.data = '../data';
dirs.results = 'interm_data';
load(fullfile(dirs.data,'AllData.mat'));
AllData_Expt2 = load(fullfile(dirs.data,('AllData_Expt2.mat')));
addpath('models');

run_BO_stock = 1;
run_fit_self = 1;

%% Subjects
Sub = [101 102 103 104 105 106 107 108 109 110 112 113 114 115 116 118 119 120 121 122 123 124 125 126 127 128];
Sub2 = [101:130];
nSub = length(Sub);
nSub2 = length(Sub2);
%% Colors
% Original
col_code(1,:) = [0.2980392156862745, 0.4470588235294118, 0.6901960784313725];
col_code(2,:) = [0.3333333333333333, 0.6588235294117647, 0.40784313725490196];
col_code(3,:) = [0.7686274509803922, 0.3058823529411765, 0.3215686274509804];


% col_code(1,:) = [0.2823529411764706, 0.47058823529411764, 0.8117647058823529];
% col_code(2,:) = [0.41568627450980394, 0.8, 0.396078431372549];
% col_code(3,:) = [0.8392156862745098, 0.37254901960784315, 0.37254901960784315];

err_col(1,:) = [0.4, 0.6, 0.9];
err_col(2,:) = [0.4, 0.8, 0.5];
err_col(3,:) = [0.9, 0.4, 0.4];
font_size = 10;

%% Generating interim files
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


%% Create Figure
figure(1)  
set(gcf,'Position',[0 0 500 700]);

%% Panel A: Subject 3 Phase 1 Data
subplot(21,20,[1:17,21:37,41:47]);
load (fullfile(dirs.results,'BayesianStockPs.mat'));

i = 3; % Model Subject
sub = num2str(Sub(i));
set(gca,'FontSize',font_size)
hold on
VT = AllData{i,3}.NoAdvice{1,1}.ValidTrials;

plot(AllData{i,3}.NoAdvice{1,1}.pUP(VT),'LineWidth',2,'Color',col_code(1,:),'LineStyle','--');
plot([0,100],[0.5,0.5],'Color','k','LineStyle','--');
plot(fit_p{i,2}.pUP(1:100),'LineWidth',2.5,'Color',col_code(2,:));

ylabel('p(UP) ');
xlabel('Trial ');
axis([0 100 0 1]);
% legend(' p(UP)','','','outcome','','Location','eastoutside');
% legend boxoff    


%plotOutcome
thisOutcome = AllData{i,3}.NoAdvice{1,1}.StockOutcome;
temp = thisOutcome < 0;
thisOutcome(temp) = 0;
scatter(find(VT),thisOutcome(VT),18,col_code(3,:),'fill','o');

%% PanelB: Number of trials predicted 

input_file = fullfile(dirs.results,'Fits_Self_Latents.mat');
load(input_file);

for s = 1:nSub
    this_choiceprob = latents{s,1}.choice_prob;
    pctCorrect(s,1) = sum(this_choiceprob > 0.5)/100;
end

subplot(21,2,[9,11,13,15,17]);
% subplot(7,10,[21

hold on
hist(pctCorrect,8);
%scatter(1:nSub,sort(pctCorrect),[],[0 0 0],'fill');
axis([0.3 0.9 0 7]);
h = findobj(gca,'Type','patch');
set(h,'FaceColor',[0 .5 .5],'EdgeColor','w')

set(gca,'FontSize',font_size)
ylabel('Participants ');
xlabel('Proportion of trials predicted ');
plot([0.485,0.485],[0,7],'Color','k','LineStyle','--','LineWidth',2);


%% Panel C
bins = [0:0.1:1];
obs_cp = NaN(nSub,length(bins)-1);

for s = 1:nSub
    thisData = AllData{s,3}.NoAdvice{1,1}.Choice;
    thislatent = latents{s,1}.pUP;
    
    [bincount,indc] = histc(thislatent,bins);
    
    for i = 1:length(bins)-1
        obs_cp(s,i) = nanmean(thisData(find(indc == i)));
    end
    
end

subplot(21,2,[10,12,14,16,18]);
hold on

avg_obs_cp(1,:) = nanmean(obs_cp);
avg_obs_cp(2,:) = nanstd(obs_cp)./sqrt(sum(~isnan(obs_cp)));

hold on
set(gca,'FontSize',font_size)
h = errorbar([1:10]-0.5,avg_obs_cp(1,:),avg_obs_cp(2,:));
set(h,'Color',[0 0 0],'linestyle','none','Marker','.','MarkerSize',20);
plot([0,10],[0,1],'Color','k','LineStyle','--');
set(gca,'ytick',[0.1:0.1:1]);
set(gca,'xtick',[1:10],'xticklabel',[0.1:0.1:1]);
axis([0 10 0 1]);
ylabel('Observed Choice Frequency ');
xlabel(sprintf('Model Predicted Choice Frequency'));

