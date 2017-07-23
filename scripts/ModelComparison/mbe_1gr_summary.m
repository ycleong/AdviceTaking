function summary = mbe_1gr_summary(mcmcChain,compVal)
%% mbe_1gr_summary
% NOTE: This was taken from the MBE Toolbox written by Nils Winter
%   Included here to allow for readers to reproduce the analyses from "Leong
%   & Zaki, Unrealistic Optimism in Advice Taking". Would be happy to take
%   this down upon request from the original author.
% 
% Computes summary statistics for all parameters of a one group estimation.
%   This will only work for a mcmc chain with parameters mu1,sigma1,
%   and nu.
%
% INPUT:
%   mcmcChain
%       structure with fields for mu, sigma, nu
%   compVal
%       comparison value. Needed to compute effect size.
%
% OUTPUT:
%   summary
%       outputs structure containing mu1, sigma1,
%       nu, nuLog10 and effectSize
%
% EXAMPLE:
%   summary = mbe_1gr_summary(mcmcChain);

% Largely based on R code introduced in the following paper:
% Kruschke, J.K., Bayesian Estimation supersedes the t-test.
% Journal of Experimental Psychology: General, Vol 142(2), May 2013, 573-603. 
% see http://www.indiana.edu/~kruschke/BEST/ for R code
% Nils Winter (nils.winter1@gmail.com)
% Johann-Wolfgang-Goethe University, Frankfurt
% Created: 2016-04-25
% Version: v1.0 (2016-04-25)
%-------------------------------------------------------------------------
summary.mu = mbe_summary(mcmcChain.mu1);
summary.sigma = mbe_summary(mcmcChain.sigma1);
summary.nu = mbe_summary(mcmcChain.nu1);
summary.nuLog10 = mbe_summary(log10(mcmcChain.nu1));
effSzChain = (mcmcChain.mu1 - compVal)./mcmcChain.sigma1;
summary.effSz = mbe_summary(effSzChain,0);
end

