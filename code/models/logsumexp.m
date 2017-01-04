function s = logsumexp(b)
    % s = logsumexp(b) by Tom Minka
    B = max(b);
    b = b - B;
    s = B + log(sum(exp(b)));