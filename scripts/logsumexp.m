% s = logsumexp(b) by Tom Minka
% Faster function to compute log sum exponent
function s = logsumexp(b)
    B = max(b);
    b = b - B;
    s = B + log(sum(exp(b)));