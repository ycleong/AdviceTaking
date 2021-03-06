%% Bayesian Learner
% Implementation for the Bayesian Learning model.
% Code is adapted from Michael Waskom's probability learner (used in Waskom, Frank, & Wagner, 2016):
%   https://github.com/mwaskom/optlearner/blob/master/ProbabilityLearner.ipynb
% Which was in turn adapted from Tim Behren's model in Behrens et al., 2007:
%   https://www.ncbi.nlm.nih.gov/pubmed/17676057
% Inputs -
%     thisOutcome: vector of [1,0] indicating history of advisor performance
%     betaprior: prior distribution 
%     alpha: not used in this model
% Outputs -
%     fit_p: data structure containing posterior distribution on each trial, as well as the mean
%     posterior estimate
% YC Leong 7/21/2017
function [fit_p] = NoCB_Learner(thisOutcome,betaprior,alpha)

% Set up parameter grids
p_grid = [0.01:0.01:0.99];
I_grid = [log(2):0.1:log(10000)];

% Set up the transitional distribution on p
[p, p_p1, I_p1] = meshgrid(p_grid,p_grid,I_grid);

a = 1 + exp(I_p1) .* p;
b = 1 + exp(I_p1) .* (1 - p);

logkerna = (a - 1) .* log(p_p1);
logkernb = (b - 1) .* log(1 - p_p1);
betaln_ab = gammaln(a) + gammaln(b) - gammaln(a + b);
p_trans = exp(logkerna + logkernb - betaln_ab);
norm_constant = repmat(sum(p_trans,1),99,1);
p_trans = p_trans./norm_constant;

% Reset()
% Initialize joint distribution
joint_dist(1,:,:) = ones(length(p_grid),length(I_grid));
%size(joint_dist)
%size(repmat(betaprior,1,86))
joint_dist(1,:,:) = squeeze(joint_dist(1,:,:)) .* repmat(betaprior,1,86);

joint_dist(1,:,:) = joint_dist/sum(joint_dist(:));

% Load data
thisOutcome(thisOutcome < 1) = 0;
nTrials = length(thisOutcome);

fit_p.p_dist = NaN(nTrials+1,99);
fit_p.pUP = NaN(nTrials+1,1);

pI = p_trans .* repmat(joint_dist(1,:,:),99,1);
pI = squeeze(sum(pI,2));

fit_p.p_dist(1,:) = sum(pI,2);
fit_p.pUP(1) = p_grid * sum(pI,2);
    
% Fit data
for i = 1:length(thisOutcome)
    if isnan(thisOutcome(i))
        fit_p.pUP(i+1) = fit_p.pUP(i);
    else
    % Multiply P(p_p+1 | p_i, I) by P(p_i, I) and integrate out p_i, which gives P(p_i+1, I)
    pI = p_trans .* repmat(joint_dist(1,:,:),99,1);
    pI = squeeze(sum(pI,2));
    

    % Update P(p_i+1, I) based on the newly observed data
    if thisOutcome(i)
        lik = p_grid;
    else
        lik = 1-p_grid;
    end
    
    % Implement confirmation bias
%     if fit_p.pUP(i) > 0.5
%         bias = (fit_p.pUP(i) - 0.5)^alpha;        
%         lik = (1-bias) * lik +  bias * p_grid;
%         
%     else
%         bias = (0.5 - fit_p.pUP(i))^alpha;
%         lik = (1-bias) * lik + bias * (1-p_grid);
%     end
    
    pI = pI .* repmat(lik',1,86);
    pI = pI ./ sum(pI(:));

    joint_dist(1,:,:) = pI;
 
    fit_p.p_dist(i+1,:) = sum(squeeze(joint_dist(1,:,:)),2);
    fit_p.pUP(i+1) = p_grid * sum(squeeze(joint_dist(1,:,:)),2);
    end
end




