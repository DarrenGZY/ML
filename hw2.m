clear;
clc;
load('~/Downloads/hw2_data_mat/mnist_mat.mat');
%% initialize
T = 100;
w = zeros(15,T+1);
delta = 1.5;
lambda = 1;

p_joint = zeros(1,T);

pd = makedist('Normal', 0,1);

%% (a)--------------------
for t=2:T+1
    
    x_w = (Xtrain.')*w(:,t-1);
    
    E_fi = x_w+(2*ytrain.'-1).*(delta*pdf(pd, (-x_w/delta)))./(ytrain.'+(1-2*ytrain.').*cdf(pd, (-x_w/delta)));

    w(:,t) = (lambda*eye(15)+Xtrain*(Xtrain.')/(delta.^2))\(sum(bsxfun(@times,Xtrain,(E_fi.')),2)/(delta.^2));
    
    x_w2 = (Xtrain.')*w(:,t);
    
    p_joint(t) = 0.5*log(lambda/2*pi)-0.5*lambda*(w(:,t).')*w(:,t)+sum((ytrain.').*log(cdf(pd, x_w2/delta))) ...
            +sum((1- ytrain.').*log(1-cdf(pd, x_w2/delta)));

end
%% (b)
figure();
plot(1:1:T,p_joint(2:T+1));


%% (c)

y_pred_prob = cdf(pd, (Xtest.')*w(:,T+1)/delta);
y_pred = double(y_pred_prob > 0.5);
y_pred = y_pred.';

C4_4=size(find(~y_pred(find(~ytest))),2);
C4_9=size(find(y_pred(find(~ytest))),2);
C9_4=size(find(~y_pred(find(ytest))),2);
C9_9=size(find(y_pred(find(ytest))),2);

confusion_matrix=[C4_4,C4_9;C9_4,C9_9];

%% (d)
index_C4_9=find(y_pred(find(~ytest)));
index_C9_4=find(~y_pred(find(ytest)));

image_C4_9_0=transpose(reshape(Q*Xtest(:,index_C4_9(1)),[28,28]));
image_C4_9_1=transpose(reshape(Q*Xtest(:,index_C4_9(2)),[28,28]));
image_C9_4_0=transpose(reshape(Q*Xtest(:,index_C9_4(1)),[28,28]));

pre_C4_9_0= y_pred_prob(index_C4_9(1));
pre_C4_9_1= y_pred_prob(index_C4_9(2));
pre_C9_4_0= y_pred_prob(index_C9_4(1));

figure();
imagesc(image_C4_9_0);
figure();
imagesc(image_C4_9_1);
figure();
imagesc(image_C9_4_0);

%% (e)
dissim = abs(y_pred_prob - 0.5);
[dissim,index_org] = sort(dissim);

image_ambiguous_1=transpose(reshape(Q*Xtest(:,index_org(1)),[28,28]));
image_ambiguous_2=transpose(reshape(Q*Xtest(:,index_org(2)),[28,28]));
image_ambiguous_3=transpose(reshape(Q*Xtest(:,index_org(3)),[28,28]));

pre_amb_1 = y_pred_prob(index_org(1));
pre_amb_2 = y_pred_prob(index_org(2));
pre_amb_3 = y_pred_prob(index_org(3));

figure();
imagesc(image_ambiguous_1);
figure();
imagesc(image_ambiguous_2);
figure();
imagesc(image_ambiguous_3);
%% (f)
w_1=transpose(reshape(Q*w(:,2),[28,28]));
w_5=transpose(reshape(Q*w(:,6),[28,28]));
w_10=transpose(reshape(Q*w(:,11),[28,28]));
w_25=transpose(reshape(Q*w(:,26),[28,28]));
w_50=transpose(reshape(Q*w(:,51),[28,28]));
w_100=transpose(reshape(Q*w(:,101),[28,28]));

figure();
imagesc(w_1);
figure();
imagesc(w_5);
figure();
imagesc(w_10);
figure();
imagesc(w_25);
figure();
imagesc(w_50);
figure();
imagesc(w_100);