function [lik,latents] = basic_bayes4(pHat,choice,FitParms,X,a1,b1,a2,b2,a3,b3,a4,b4)

Beta = X(1);
lik = 0;
latents.pUP = NaN(length(choice),1);
latents.choice_prob = NaN(length(choice),1);

for t = 1:length(choice)
    if ~isnan(choice(t))
        pUP = 1/(1+exp(-Beta*(pHat(t)-0.5)));
        
        if choice(t)
            choice_prob = pUP;
        else
            choice_prob = 1-pUP;
        end
        
        lik = lik + log(choice_prob);
    
        latents.pUP(t) = pUP;
        latents.choice_prob(t) = choice_prob;
    end
end

lik = -lik;

for i = 1
    if (FitParms.Use(1))    % putting a Gamma prior on Beta
        lik = lik - log(gampdf(X(1),FitParms.Parms(1,1),FitParms.Parms(1,2)));
    end
    
    if (FitParms.Use(2))    % putting a Gamma prior on Beta
        lik = lik - log(exppdf(a1,FitParms.Parms(2,1)));
        lik = lik - log(exppdf(a2,FitParms.Parms(2,1)));
        lik = lik - log(exppdf(a3,FitParms.Parms(2,1)));
        lik = lik - log(exppdf(a4,FitParms.Parms(2,1)));
        lik = lik - log(exppdf(b1,FitParms.Parms(2,1)));
        lik = lik - log(exppdf(b2,FitParms.Parms(2,1)));
        lik = lik - log(exppdf(b3,FitParms.Parms(2,1)));
        lik = lik - log(exppdf(b4,FitParms.Parms(2,1)));
    end
end
end