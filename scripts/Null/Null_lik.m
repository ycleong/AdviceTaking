%% Null Modle Likelihood function
function [lik,latents] = Null_lik(AdvisorCorrect,choice,FitParms,X,this_model,n_advisors)

Beta = X(1);

lik = 0;
latents.choice_prob = NaN(length(AdvisorCorrect),n_advisors);
latents.pFOR = NaN(length(choice),n_advisors);
latents.choice_prob = NaN(length(choice),n_advisors);

for a = 1:n_advisors
    
    for t = 1:length(AdvisorCorrect)
        
        if ~isnan(choice(t,a))
            
            pFOR = 1/(1+exp(-Beta));
            
            if choice(t,a)
                choice_prob = pFOR;
            else
                choice_prob = 1-pFOR;
            end
            
            lik = lik + log(choice_prob);
            latents.pFOR(t,a) = pFOR;
            latents.choice_prob(t,a) = choice_prob;
            
        end
        
    end
end


% Add regulatory prirs
lik = -lik;

% Add regulatory priors
if (FitParms.Use(1))
    lik = lik - log(gampdf(Beta,FitParms.Parms(2,1),FitParms.Parms(2,2)));
end

latents.choice_prob = reshape(latents.choice_prob,length(AdvisorCorrect)*n_advisors,1);