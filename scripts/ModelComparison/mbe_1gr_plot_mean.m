function mbe_1gr_plot_mean(y, mcmcChain, compVal, varargin)
% NOTE: This was taken from the MBE Toolbox written by Nils Winter, adapted
% to give the mean instead of the median of the posterior distribution
%   Included here to allow for readers to reproduce the analyses from Leong
%   & Zaki, Unrealistic Optimism in Advice Taking. Would be happy to take
%   this down upon request from the original author.
% 
% mbe_makePlots
%   Make histogram of data with superimposed posterior prediction check
%   and plots posterior distribution of monitored parameters.
%
% INPUT:
%   y
%       cell array containing vectors for y1
%   mcmcChain
%       structure with one MCMC-chain, should contain all monitored parameters
%   compVal
%       comparison value for computing of effect size
%
% Specify the following name/value pairs for additional plot options:
%        Parameter      Value
%       'plotPairs'     show correlation plot of parameters ([1],0)
%
%
% EXAMPLE:

% Largely based on R code introduced in the following paper:
% Kruschke, J.K., Bayesian Estimation supersedes the t-test.
% Journal of Experimental Psychology: General, Vol 142(2), May 2013, 573-603. 
% see http://www.indiana.edu/~kruschke/BEST/ for R code
% Nils Winter (nils.winter1@gmail.com)
% Johann-Wolfgang-Goethe University, Frankfurt
% Created: 2016-04-25
% Version: v1.00 (2016-04-25)
%-------------------------------------------------------------------------

% -----------------------------------------------------------------
% Get input
% -----------------------------------------------------------------
p = inputParser;
defaultPlotPairs = 1;
addOptional(p,'plotPairs',defaultPlotPairs);
parse(p,varargin{:});
plotPairs = p.Results.plotPairs;

% Get parameter names
names = fieldnames(mcmcChain);

%% -----------------------------------------------------------------
% Plot correlations between parameters
%-----------------------------------------------------------------
if plotPairs
    mbe_plotPairs(mcmcChain,1000)
end

%% -----------------------------------------------------------------
% Plot data y and smattering of posterior predictive curves:
%-----------------------------------------------------------------
nu = mcmcChain.(names{3});
mu = mcmcChain.(names{1});
sigma = mcmcChain.(names{2});
figure('Color','w','NumberTitle','Off','Position',[100,50,800,600]);
subplot(3,2,[2 4]);
mbe_plotData(y,nu,mu,sigma);

%% -----------------------------------------------------------------
% Plot posterior distribution of parameter nu:
%-----------------------------------------------------------------
subplot(3,2,6);
mbe_plotPost(log10(nu),'credMass',0.95,'xlab','log10(\nu)','PlotTitle','Normality');

%-----------------------------------------------------------------
% Plot posterior distribution of parameters mu:
%-----------------------------------------------------------------
xLim(1) = min(mu);
xLim(2) = max(mu);

subplot(3,2,1);
mbe_plotPost(mu,'xlab','\mu','xlim',xLim,'Plottitle','Mean','showMode',0);

%-----------------------------------------------------------------
% Plot posterior distribution of param's sigma1, sigma2, and their difference:
%-----------------------------------------------------------------
xLim(1) = min(sigma);
xLim(2) = max(sigma);
subplot(3,2,3);
mbe_plotPost(sigma,'xlab','\sigma','xlim',xLim,'PlotTitle','Std. Dev.','showMode',0);

%-----------------------------------------------------------------
% Plot of estimated effect size. Effect size is d-sub-a from
%-----------------------------------------------------------------
% Macmillan & Creelman, 1991; Simpson & Fitter, 1973; Swets, 1986a, 1986b.
effectSize = (mu - compVal) ./ sigma;
subplot(3,2,5);
str = ['(\mu-' num2str(compVal) ')/\sigma'];
mbe_plotPost(effectSize,'rope',[-0.1,0.1],'xlab',str,'PlotTitle','Effect Size','showMode',0);


end
