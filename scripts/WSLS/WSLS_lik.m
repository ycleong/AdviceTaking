%% WSLS Likelihood function
function [lik,latents] = WSLS_lik(AdvisorCorrect,Choice,FitParms,X,this_model,n_advisors)

pSW = X(1);
pSL = X(2);

lik = 0;
latents.choice_prob = NaN(length(AdvisorCorrect),3);


for a = 1:n_advisors
    for t = 1:length(AdvisorCorrect)
        
        if ~isnan(Choice(t,a))
            
            % Assign 0.5 to first trial
            if t == 1
                lik = lik + log(0.5);
                latents.choice_prob(t,a) = 0.5;
            elseif isnan(Choice(t-1,a))
                latents.choice_prob(t,a) = NaN;               
            else
                LastAdvisorCorrect = AdvisorCorrect(t-1,a);
                LastChoice = Choice(t-1,a);
                thisChoice = Choice(t,a);
                
                % Check if participant won last round
                if LastAdvisorCorrect == LastChoice % Win
                    % Win
                    if thisChoice == LastChoice
                        % Win Stay
                        lik = lik + log(pSW);
                        latents.choice_prob(t,a) = pSW;
                    else
                        % Lose Shift
                        lik = lik + log(1-pSW);
                        latents.choice_prob(t,a) = 1 - pSW;
                    end
                    
                else
                    % Lose
                    if thisChoice == LastChoice
                        % Lose Stay
                        lik = lik + log(1-pSL);
                        latents.choice_prob(t,a) = 1-pSL;
                    else
                        % Lose Shift
                        lik = lik + log(pSL);
                        latents.choice_prob(t,a) = pSL;
                    end
                    
                end
            end
        end
    end
end

% Add regulatory prirs
lik = -lik;
latents.choice_prob = reshape(latents.choice_prob,length(AdvisorCorrect)*n_advisors,1);