clc; 
clear;
load('~/Downloads/data_matlab/data1.mat')
%% Initialize the variational parameter
X = X';
n = 100; %for three datasets, n = 100,250, 500
d = n+1;

a0 = 1e-16;
b0 = 1e-16;
a = ones(d,1)*(a0 + 0.5);

e0 = 1;
f0 = 1;
lambda_c = e0 + n/2;

mu = zeros(d,1);
sigma = eye(d);
L = zeros(1,500);
%% Iterations
for i = 1:500
    lambda_d = f0  + 0.5 * mu'*X*X'*mu + 0.5 * y' * y + 0.5 * trace(X'*sigma*X) - y'*X'*mu;
    b = b0 * ones(d,1) + 0.5 * (mu.^2 + diag(sigma));
    k = lambda_c / lambda_d;
    sigma = pinv(k * X * X' + diag(a ./ b));
    mu = k * sigma * X * y;
    
    L(i) = sum( (a./b) .* (b - b0 * ones(d,1)) ) - (f0 - lambda_d) * k- 0.5*sum((a./b).*(mu.^2 + diag(sigma))) - 0.5*k*(y'*y - 2*y'*X'*mu + mu'*X*X'*mu+trace(X'*sigma*X));
end

%% result

%a)
figure(1)
plot(-L);

%b)
figure(2)
inv_expectation_alpha = b ./ a;
stem(inv_expectation_alpha);

%c)
inv_expectation_lambda = 1 / k ;


%d)
y_hat = X' * mu;
figure(3)
hold on
plot(z, y_hat, 'g');
scatter(z, y , 'r');
plot(z, 10*sinc(z), 'y')
