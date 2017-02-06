%% problem 1
clear; 
clc;
load('~/Downloads/data.mat');
%% initialize
x = X(1,:);
y = X(2,:);
[d, n] = size(X);


K = 10; %2, 4, 8 , 10
pi = ones(1,K)/K;
mu = rand(d, K);

Sigma = repmat(eye(d), 1, 1, K);
phi = zeros(n, K);

%% iteration
T = 100;
L = zeros(1, T);
for t = 1:T  
    temp = zeros(d, d, n);
    % E-step
    for i = 1:n      
        for j = 1:K
            phi(i,j) = pi(j) * mvnpdf(X(:,i), mu(:,j), Sigma(:,:,j));
        end
    end
    phi = phi ./ repmat(sum(phi,2), 1, K);
    
    % M-step
    num = sum(phi, 1);
    mu_x = sum(phi .* repmat(x', 1, K), 1) ./ num;
    mu_y = sum(phi .* repmat(y', 1, K), 1) ./ num;
    mu = [mu_x; mu_y];
       
    for j = 1:K
        for i = 1:n
            temp(:,:,i) = (X(:,i)-mu(:,j))*(X(:,i)-mu(:,j))' * phi(i,j);
        end
        Sigma(:,:,j) = sum(temp, 3) / num(j);
    end
    
    pi = num / n;
    
    % Log Likelihood
    tempL = zeros(n, K);
    for i = 1: n
        for j = 1: K
            tempL(i, j) = phi(i,j) * (0.5*log(det(inv(Sigma(:,:,j)))) - 0.5*(X(:,i) - mu(:,j))'*inv(Sigma(:,:,j))*(X(:,i) - mu(:,j)) + log(pi(j)));
        end
    end
    L(t) = sum(sum(tempL));
end

%% result
figure(1)
plot(L)


figure(2)
hold on
[~, index] = sort(phi, 2, 'descend');
index = index(:,1);
for i = 1: K
    plot(X(1,index == i), X(2,index == i) ,'o')
end
hold off