clear;
clc;
load('~/Downloads/hw1_data_mat/mnist_mat.mat');

%---------(a)----------
a=1;
b=1;
c=1;
e=1;
f=1;

index_y1 = find(ytrain);
index_y0 = find(~ytrain);

N0=size(index_y0, 2);
N1=size(index_y1, 2);

sumX_y0=sum(Xtrain(:,index_y0),2);
sumX_y1=sum(Xtrain(:,index_y1),2);

averageX_y0=sumX_y0/N0;
averageX_y1=sumX_y1/N1;

%student t distribution scale parameter u
t_y0_u=sumX_y0/(N0+1/a);
t_y1_u=sumX_y1/(N1+1/a);

m0=zeros(15,1);
m1=zeros(15,1);

for n=1:15
    m0(n)=c+0.5*sum((Xtrain(n,index_y0)-averageX_y0(n)).^2)+(averageX_y0(n)^2)*(N0/a)/((N0+1/a)*2);
    m1(n)=c+0.5*sum((Xtrain(n,index_y1)-averageX_y1(n)).^2)+(averageX_y1(n)^2)*(N1/a)/((N1+1/a)*2);
end

%student t distribution scale parameter delta
t_y0_delta=m0*(N0+1/a+1)/((N0+1/a)*(b+N0/2));
t_y1_delta=m1*(N1+1/a+1)/((N1+1/a)*(b+N1/2));

%t distribution freedom degree
v0=b+N0/2;
v1=b+N1/2;

predict_y0_x=ones(1,1991);
predict_y1_x=ones(1,1991);


%use tpdf function for student't t distribution
for n=1:15
    predict_y0_x=predict_y0_x.*(tpdf((Xtest(n,:)-t_y0_u(n))/sqrt(t_y0_delta(n)),v0));
    predict_y1_x=predict_y1_x.*(tpdf((Xtest(n,:)-t_y1_u(n))/sqrt(t_y1_delta(n)),v1));
end

factor_y0=(f+N0)/(e+f+N0+N1);
factor_y1=(e+N0)/(e+f+N0+N1);

pre_y0_x=predict_y0_x*factor_y0;
pre_y1_x=predict_y1_x*factor_y1;

%---------(b)----------
result=zeros(1,1991);
index=find(pre_y0_x<=pre_y1_x);
result(index)=1;

C4_4=size(find(~result(find(~ytest))),2);
C4_9=size(find(result(find(~ytest))),2);
C9_4=size(find(~result(find(ytest))),2);
C9_9=size(find(result(find(ytest))),2);

confusion_matrix=[C4_4,C4_9;C9_4,C9_9];

%---------(c)----------
index_C4_9=find(result(find(~ytest)));
index_C9_4=find(~result(find(ytest)));

image_C4_9_0=transpose(reshape(Q*Xtest(:,index_C4_9(1)),[28,28]));
image_C4_9_1=transpose(reshape(Q*Xtest(:,index_C4_9(2)),[28,28]));
image_C9_4_0=transpose(reshape(Q*Xtest(:,index_C9_4(1)),[28,28]));

pre_C4_9_0=pre_y1_x(index_C4_9(1))/(pre_y1_x(index_C4_9(1))+pre_y0_x(index_C4_9(1)));
pre_C4_9_1=pre_y1_x(index_C4_9(2))/(pre_y1_x(index_C4_9(2))+pre_y0_x(index_C4_9(2)));
pre_C9_4_0=pre_y0_x(index_C9_4(1))/(pre_y1_x(index_C9_4(1))+pre_y0_x(index_C9_4(1)));


imagesc(image_C4_9_0);
figure();
imagesc(image_C4_9_1);
figure();
imagesc(image_C9_4_0);

%---------(d)----------
dissim = abs((pre_y1_x./(pre_y0_x+pre_y1_x)) - 0.5);
[dissim,index_org] = sort(dissim);

image_ambiguous_1=transpose(reshape(Q*Xtest(:,index_org(1)),[28,28]));
image_ambiguous_2=transpose(reshape(Q*Xtest(:,index_org(2)),[28,28]));
image_ambiguous_3=transpose(reshape(Q*Xtest(:,index_org(3)),[28,28]));

pre_amb_1=pre_y1_x(index_org(1))/(pre_y1_x(index_org(1))+pre_y0_x(index_org(1)));
pre_amb_2=pre_y1_x(index_org(2))/(pre_y1_x(index_org(2))+pre_y0_x(index_org(2)));
pre_amb_3=pre_y1_x(index_org(3))/(pre_y1_x(index_org(3))+pre_y0_x(index_org(3)));

figure();
imagesc(image_ambiguous_1);
figure();
imagesc(image_ambiguous_2);
figure();
imagesc(image_ambiguous_3);