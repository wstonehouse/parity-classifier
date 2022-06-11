%% Activation function
%% Input: matrix of activation values
%% Output: sigmoid on those values
function y = activation_fn(x)
    y = 1./(1+exp(-x));
end

%% f prime function
function y = fprime(x)
    y = exp(-x)./(1+exp(-x)).^2;
end
