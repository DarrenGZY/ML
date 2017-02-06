clc; 
clear;
load('~/Downloads/data.mat')

%% initialize
K = 2;
T = 10;
N = size(X,2);

pi = zeros(K);
miu = zeros(2,K);
Sigma = zeros(2,2,K);

for i = 1:K
    pi(i) = 1/K;
    Sigma(:,:,i) = [1,0;0,1];
end


%% iteration
%E-step

for t = 1:T
    fi = zeros(2,K,N); 
    for i = 1:N
        sums = [0,0];
        for k = 1:K
            sums = sums + pi(k)*mvnpdf(X(:,i), miu(:,k), inv(Sigma(:,:,k)));
        end
        for j = 1:K
            fi(:,j,i) = pi(j)*mvnpdf(X(:,i), miu(:,j), inv(Sigma(:,:,j)))./sums;
        end
    end
    
    for j = 1:K
        n = sum(fi, 3);
        fi_x_sum = [0;0];
        for i = 1:N
            fi_x_sum = fi_x_sum + fi(:,j,i).*X(:,i);
        end
        miu(:,j) = fi_x_sum/n;
    end
end
    
    
    
    
    
    
    
    
