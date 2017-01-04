function [fit_p] = ProbabilityLearner(thisOutcome)
%% Probability Learner - Implementing Michael's code in Matlab

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
norm_constant = repmat(sum(p_trans,1),99,1,1);
p_trans = p_trans./norm_constant;

% Reset()
% Initialize joint distribution
joint_dist(1,:,:) = ones(length(p_grid),length(I_grid));
joint_dist(1,:,:) = joint_dist/sum(joint_dist(:));

% Load data
thisOutcome(thisOutcome < 1) = 0;
nTrials = length(thisOutcome);

fit_p.p_dist = NaN(nTrials+1,99);
fit_p.pUP = NaN(nTrials+1,1);

pI = p_trans .* repmat(joint_dist(1,:,:),99,1,1);
pI = squeeze(sum(pI,2));

fit_p.p_dist(1,:) = sum(pI,2);
fit_p.pUP(1) = p_grid * sum(pI,2);
    
% Fit data
for i = 1:length(thisOutcome)
    
    if isnan(thisOutcome(i))
    
    else
    % Multiply P(p_p+1 | p_i, I) by P(p_i, I) and integrate out p_i, which gives P(p_i+1, I)
    pI = p_trans .* repmat(joint_dist(1,:,:),99,1,1);
    pI = squeeze(sum(pI,2));
    
    % Update P(p_i+1, I) based on the newly observed data
    if thisOutcome(i)
        lik = p_grid;
    else
        lik = 1-p_grid;
    end
    
    pI = pI .* repmat(lik',1,86);
    pI = pI ./ sum(pI(:));
    
    joint_dist(1,:,:) = pI;
    
    fit_p.p_dist(i+1,:) = sum(pI,2);
    fit_p.pUP(i+1) = p_grid * sum(pI,2);
    
    end
    
end
end




