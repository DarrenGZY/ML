% Problem 3
clear; 
clc;
load('~/Downloads/data.mat');


%% initilize
[d, n] = size(X);

result = zeros(6, 500);
K = zeros(1, 500);

c0 = 1/10;
c = c0;
a0 = d;
alpha0 = 1;
a = a0;
A = cov(X');
B0 = c0*d*A;
B = B0;
cluster = ones(1, n);
k = 1;
m0 = mean(X,2);
m = m0;

Lambda = wishrnd(inv(B0), a0);
mu = mvnrnd(m0,inv(c0*Lambda))';


T=500;

%% update
for t =  1: T

    for i = 1:n
        phi = zeros(1,k+1);
        
        for j = 1: k
            num_index = numel(find(cluster == j));
            if cluster(i) == j
                num_index = num_index - 1;
            end
            if num_index > 0
                phi(j) = mvnpdf(X(:,i), mu(:,j), inv(Lambda(:,:,j))) * num_index / (alpha0 + n - 1);
            end
        end

        phi(k+1) = alpha0 / (alpha0 + n - 1) * (c0/(1+c0)/pi)^(d/2) * (det(B0+c0/(1+c0)*(X(:,i)-m0)*(X(:,i)-m0)'))^(-(a0+1)/2) / ((det(B0))^(-a0/2)) * gammaHelper(a0, d);
        phi = phi ./ sum(phi);
        temp_index = randsample([1:k+1],1,true,phi);
        cluster(i) = temp_index;
        if cluster(i) == k + 1           
            nij = tabulate(cluster);
            s = nij(:,2);
            temp_index = find(cluster == k+1);
            temp_sum = sum(X(:,temp_index),2);
            m(:,k+1) = c0/(c0+s(k+1)) * m0 + 1/(c0+s(k+1)) * temp_sum;
            c(k+1) = s(k+1) + c0;
            a(k+1) = a0 + s(k+1);
            xbar = mean(X(:,temp_index), 2);
            B(:,:,k+1) = B0 + c0*s(k+1)/(c0+s(k+1)) * (xbar - m0)*(xbar - m0)' + (X(:,temp_index)-repmat(xbar, 1, s(k+1)))*(X(:,temp_index)-repmat(xbar, 1, s(k+1)))';   
            Lambda(:,:,k+1) = wishrnd(inv(B(:,:,k+1)), a(k+1));
            mu(:,k+1) = mvnrnd(m(:,k+1),inv(c(k+1)*Lambda(:,:,k+1)));
            k = k + 1;
        end
    end
    %re-index
    tab = tabulate(cluster);
    keep = find(tab(:,2) >= 1);
    tab = tab(keep,:);
    k = size(tab,1);
    for j=1:k
        temp_cluster(find(cluster == tab(j,1))) = j;
    end
    cluster = temp_cluster;
    nij = tabulate(cluster);
    s = nij(:,2);
    %re-sample
    for j = 1: k
        temp_index = find(cluster == j);
        temp_sum = sum(X(:,temp_index),2);
        m(:,j) = c0/(c0+s(j)) * m0 + 1/(c0+s(j)) * temp_sum;
        c(j) = s(j) + c0;
        a(j) = a0 + s(j);
        xbar = mean(X(:,temp_index), 2);
        B(:,:,j) = B0 + c0*s(j)/(c0+s(j)) * (xbar - m0)*(xbar - m0)' + (X(:,temp_index)-repmat(xbar, 1, s(j)))*(X(:,temp_index)-repmat(xbar, 1, s(j)))';
        Lambda(:,:,j) = wishrnd(inv(B(:,:,j)), a(j));
        mu(:,j) = mvnrnd(m(:,j),inv(c(j)*Lambda(:,:,j)));
    end 
    temp_result = sort(s,'descend')';
    length = numel(temp_result);
    if length >= 6
        length = 6;
    end
    result(1:length,t) = temp_result(1:length);
    K(t) = k;
end

figure(1)
hold on
for i = 1:6
    plot(result(i,:))
end
hold off

figure(2)
plot(K)