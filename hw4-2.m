%% problem 2
clear; 
clc;
load('~/Downloads/data.mat');
%% initialize
x = X(1,:);
y = X(2,:);
[d, n] = size(X);


K = 2; %2, 4, 10, 25
alpha0 = 1;
alpha = 1 * ones(1, K);
A = cov(X');
B0 = d / 10 * A;
B = repmat(eye(d), 1, 1, K);
c = 10;
a0 = d;
a = rand(1, K);
m = rand(2, K);
m0 = mean(X,2);
% m = X(:,1:K);
Sigma = repmat(eye(d), 1, 1, K);

%% iteration
T = 100;
L = zeros(1, T);
phi = zeros(n, K);
for t = 1: T
    for i = 1:n
        for j = 1:K
            t1 = digamma(a(j) / 2) + digamma(a(j)/2 - 0.5) - log(det(B(:,:,j)));
            t2 = (X(:,i)-m(:,j))'*(a(j)*inv(B(:,:,j)))*(X(:,i)-m(:,j));
            t3 = trace(a(j)*inv(B(:,:,j))*Sigma(:,:,j));
            t4 = digamma(alpha(j)) - digamma(sum(alpha));
            phi(i,j) = exp(0.5*t1 - 0.5*t2- 0.5*t3 + t4);
        end
    end
    phi = phi ./ repmat(sum(phi,2), 1, K);
    num = sum(phi, 1);
    alpha = alpha0 * ones(1, K) + num;
    
    for j = 1:K
        Sigma(:,:,j) = inv(1/c * eye(2) + num(j)*a(j)*inv(B(:,:,j)));
        x_temp = x * phi(:,j);
        y_temp = y * phi(:,j);
        xx = [x_temp; y_temp];
        m(:,j) = Sigma(:,:,j) * (a(j)*inv(B(:,:,j))*xx);
    end
    
    a = a0 * ones(1, K) + num;
    for j = 1:K
        temp = 0;
        for i = 1:n
            temp = temp + phi(i,j)*((X(:,i)-m(:,j))*(X(:,i)-m(:,j))' + Sigma(:,:,j));
        end
        B(:,:,j) = B0 + temp;
    end
    for j = 1:K
        t1 = digamma(a(j) / 2) + digamma(a(j)/2 - 0.5) - log(det(B(:,:,j)));
        t2 = (X(:,i)-m(:,j))'*(a(j)*inv(B(:,:,j)))*(X(:,i)-m(:,j));
        t3 = trace(a(j)*inv(B(:,:,j))*Sigma(:,:,j));
        
        L(t) = L(t) + 0.5*(num(j)*(t1-t3-t2)) - 0.5*(t1-1/c*(trace(Sigma(:,:,j))+m(:,j)'*m(:,j))-a(j)*trace(B0*inv(B(:,:,j))));
        L(t) = L(t) + (- 0.5*(a(j)-3)*t1 + a(j)*(1+log(2)) - a(j)/2*log(det(B(:,:,j))) + log(gammad(a(j)/2, 2)) + 0.5*det(Sigma(:,:,j)));
    end
    temp = 0;
    for i = 1:n
        for j= 1:K
            t4 = digamma(alpha(j)) - digamma(sum(alpha));
            temp = phi(i,j) * (log(t4) + log(phi(i,j)));
        end
    end
    L(t) = L(t) + temp;
end

figure(1)
plot(1:100, L);

figure(2)
hold on
[~, index] = sort(phi, 2, 'descend');
index = index(:,1);
for i = 1: K
    plot(X(1,index == i), X(2,index == i) ,'o')
end
hold off





