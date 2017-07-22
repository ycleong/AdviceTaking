%% RL Likelihood function
function [lik,latents] = RL_lik(AdvisorCorrect,choice,FitParms,X,this_model,n_advisors)

AdvisorCorrect(AdvisorCorrect == 0) = -1;

Eta = X(1);
Beta = X(2);
V_init = 0;

lik = 0;

if nargout == 2    
    latents.V = NaN(length(choice),3);
    latents.pFOR = NaN(length(choice),3);
    latents.choice_prob = NaN(length(choice),3);
    latents.PE = NaN(length(choice),3);
end

for a = 1:n_advisors
    
    V = V_init;
    
    for t = 1:length(AdvisorCorrect)
        
        if ~isnan(choice(t,a))
            
            pFOR = 1/(1+exp(-Beta*V));
            
            if choice(t,a)
                choice_prob = pFOR;
            else
                choice_prob = 1-pFOR;
            end
            
            lik = lik + log(choice_prob);
            
            if nargout == 2
                latents.V(t,a) = V;
                latents.pFOR(t,a) = pFOR;
                latents.choice_prob(t,a) = choice_prob;
            end
            
            PE = AdvisorCorrect(t,a) - V;
            V = V + Eta*PE;
            
            if nargout == 2
                latents.PE(t,a) = PE;
            end
        end
        
    end
end

% Calculate negative log likelihood
lik = -lik;

% Add regulatory priors
if (FitParms.Use(2))
    lik = lik - log(gampdf(Beta,FitParms.Parms(2,1),FitParms.Parms(2,2)));
end

if nargout == 2
    latents.choice_prob = reshape(latents.choice_prob,length(AdvisorCorrect)*n_advisors,1);
end