%% Sample code
% Compares the performance of recursiveNystrom to uniform Nystrom and 
% on a forest covertype dataset.

% download forest covertype data from UCI repository
covtype = websave('covtype.csv.gz','https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz')
covtype = gunzip(covtype)

% read in data matrix and select small subset to experiment with
A = csvread('covtype.csv');
n = 2000;
% trim off last column of labels
D = A(randperm(size(A,1),n),1:end-1);

% mean center and normalize numerical data, scale binary data by 1/sqrt(n)
d = size(D,2);
means = zeros(1,d);
means(1:10) = mean(D(:,1:10));
D = D - repmat(means,n,1);
cnorms = 1/sqrt(n)*ones(1,d);
cnorms(1:10) = 1./sqrt(sum(D(:,1:10).^2,1));
cnorms(find(cnorms == Inf)) = 0;
D = D*diag(cnorms);

% set kernel variance
gamma = 256;
% explicitly construct full kernel for evaluating error
K = gaussianKernel(D,1:n,1:n,gamma);

%% Compare methods for approximating K
% these are the sample sizes we'll try
svals = [100:100:2000];
trials = 5;

%% Recursive Ridge Leverage Score Nystrom
errorsRN = zeros(trials,length(svals));
for j = 1:trials
    fprintf('Beginning Trial %d for Recursive Nystrom\n',j);
    for i = 1:length(svals)
        s = svals(i);
        fprintf('Constructing Nystrom approximation with %d samples\n',s);
        kFunc = @(D,rowInd,colInd) gaussianKernel(D,rowInd,colInd,gamma);
        [C,W] = recursiveNystrom(D,s,kFunc);
        % error is the top eigenvalue of the difference between our approximation and K
        errorsRN(j,i) = eigs(@(x) K*x - C*(W*(C'*x)), n, 1);
    end
end

%% Uniform Nystrom
errorsUniform = zeros(trials,length(svals));
for j = 1:trials
    fprintf('Beginning Trial %d for Uniform Nystrom\n',j);
    for i = 1:length(svals)
        s = svals(i);
        fprintf('Constructing Nystrom approximation with %d samples\n',s);
        kFunc = @(D,rowInd,colInd) gaussianKernel(D,rowInd,colInd,gamma);
        samp = randperm(n,s);
        C = kFunc(D,1:n,samp);
        SKS = C(samp,:);
        W = inv(SKS+(10e-6)*eye(s,s));
        % error is the top eigenvalue of the difference between our approximation and K
        errorsUniform(j,i) = eigs(@(x) K*x - C*(W*(C'*x)), n, 1);
    end
end

%% compare errors to number of samples
figure();
plot(svals,mean(abs(errorsRN),1),svals,mean(abs(errorsUniform),1));
legend('Recursive Nystrom','Uniform Nystrom')
xlabel('number of samples') 
ylabel('spectral norm error')
set(gca,'yscale','log');
ylim([min(min(mean(abs(errorsRN))),min(mean(abs(errorsUniform))))/2 max(max(mean(abs(errorsRN))),max(max(abs(errorsUniform))))*2])