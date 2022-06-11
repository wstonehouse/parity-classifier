clear;
close all;
clc;

%% Parameters of the model

lrate = 1; % learning rate
inputs = 8; % amount of input units
hidden = 3; % number of hidden units
output = 1; % number of output units
count = 8; % number of inputs/patterns

%% Generate the patterns that the model will be trained on

f = zeros(inputs, count); % allocate space for inputs

% create input pattern
for i=1:count
    f(:,i) = round(rand(inputs,1));
end

%% Determine the desired output for each pattern
%% Output a 1 if the number of ones in the input is even
%% Output a 0 if the number of ones in the input is odd

h = zeros(1,count); % allocate space for desired output vector

% create desired output
for i=1:count
    if mod(sum(f(:,i)),2) == 1
        h(i) = 1;
    else
        h(i) = 0;
    end
end

% Matrix for the weights that connect the input to hidden units
w_fg = (rand(hidden,inputs)-0.5);
% Matrix for the hidden units to the output unit
w_gh = (rand(output,hidden)-0.5);



%% Iterate through the model until one of two conditions is met:
% - the sum of squared error between the observed and desired output is less than .01
% - the number of epochs is greater than 1000 (an epoch is similar to an iteration, except that one epoch is one pass through all the input patterns)

epochs = 1000;
sse_list = [];
i=0;

%f,w_fg,activate,w_gh,activate
while i ~= epochs
    i = i+1;
    % Pass the activation from the input units (which is simply the input pattern) to the hidden units
    input_to_hidden = w_fg*f;
    % Determine hidden unit activation by passing the input to the hidden units through the activation function
    hidden_activation = activation_fn(input_to_hidden);
    % Pass the activation from the hidden units to the output units
    input_to_output = w_gh * hidden_activation;
    % Determine output activity by passing the input to the output units through the activation function
    output_activation = activation_fn(input_to_output);
    % Compute the output error
    desired_output = h;
    output_error = desired_output - output_activation;
    for j=1:8
        % Determine weight changes for each layer: w_fg, w_gh
        dw_fg = lrate*diag(fprime(w_fg*f(:,j)))*w_gh.'*diag(output_error(j))*fprime(w_gh*hidden_activation(:,j))*f(:,j).';
        dw_gh = lrate*diag(fprime(w_gh*hidden_activation(:,j)))*output_error(j)*hidden_activation(:,j).';
        % Apply the weight changes
        w_fg = w_fg + dw_fg;
        w_gh = w_gh + dw_gh;
        % Compute the sum of squared errors (SSE) over all input patterns for the current epoch
        sse = trace(output_error'*output_error);
        sse_list(end+1) = sse;
    end
                    
    % Print a brief report every 10 epochs, listing the epoch number and the SSE value
    disp('Epoch '+string(i));
    disp('SSE: '+string(sse));
    if (sse<0.01)
        plot(sse_list);
        break;
    end
end

if (i==1000)
    disp('Your model did not converge. Run again.');
    plot(sse_list);
end
                    
                    
% Let's now test the model’s generalization abilities by creating a new set of patterns, and running those patterns through the model

f = zeros(inputs, count); % allocate space for inputs

% create input pattern
for i=1:count
    f(:,i) = round(rand(inputs,1));
end

h = zeros(1,count); % allocate space for desired output vector

% create desired output
for i=1:count
    if mod(sum(f(:,i)),2) == 1
        h(i) = 1;
    else
        h(i) = 0;
    end
end

input_to_hidden = w_fg*f;

hidden_activation = activation_fn(input_to_hidden);

input_to_output = w_gh * hidden_activation;

output_activation = activation_fn(input_to_output);

desired_output = h;
output_error = desired_output - output_activation;
