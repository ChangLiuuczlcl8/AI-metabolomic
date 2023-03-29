%% NMR IBS
% replace water region with noise
fulldata = importdata('data/IBS/IBS_267_afterprocessing_class_and_NMR.csv');
extrappm = linspace(4.6798,4.8803,402);
extrappm = extrappm(2:401);
noisyregion = fulldata(2:end,2:301);
x = randi(size(noisyregion,2),1,400);
newintensities = noisyregion(:,x);
toadd = [extrappm; newintensities];
newdata = [fulldata(:,1:8361) toadd fulldata(:,8362:end)];
% remove the negative offset from each spectrum
for i = 2:size(newdata, 1)
   newdata(i, 2:end) = newdata(i, 2:end) - min(newdata(i, 2:end)); 
end
writematrix(newdata,'data/IBS/IBS267wnoise_replace.csv');

% load data
data = importdata('data/IBS/IBS267wnoise_replace.csv');
ppm = data(1,2:end);
labels = data(2:end,1);
data = data(2:end,2:end);

% plot IBS data
subplot(2,2,1);
plot(ppm,data(labels==0,:))
xlabel('chemical shift (ppm)')
ylabel('Intensity')
title('spectra from control samples')
set(gca, 'XDir','reverse')
ylim([-0.5*10e6 10e7])
subplot(2,2,2);
plot(ppm,data(labels==1,:))
xlabel('chemical shift (ppm)')
ylabel('Intensity')
title('spectra from IBS samples')
set(gca, 'XDir','reverse')
ylim([-0.5*10e6 10e7])
subplot(2,2,3);
plot(ppm,mean(data(labels==0,:),1))
xlabel('chemical shift (ppm)')
ylabel('Intensity')
title('average spectrum from control samples')
set(gca, 'XDir','reverse')
ylim([0 3*10e6])
subplot(2,2,4);
plot(ppm,mean(data(labels==1,:),1))
xlabel('chemical shift (ppm)')
ylabel('Intensity')
title('average spectrum from IBS samples')
set(gca, 'XDir','reverse')
ylim([0 3*10e6])
saveas(gcf,'output_image/NMR/data.png')

% bin to 0.005 ppm
data = reshape(data,267,10,1900);
data = sum(data,2) * 0.1;
data = reshape(data,267,1900);
ppm = reshape(sum(reshape(ppm,10,1900),1)*0.1,1900,1);

% visualisation of raw data
% PCA
zdata = zscore(data);
[~,feature_pca,~,~,explained,~] = pca(zdata);
h_pca = gscatter(feature_pca(:,1),feature_pca(:,2),labels,[],[],10);
title('PCA 2D projection of NMR raw data')
xlabel('PCA1 explained variance='+string(explained(1))) 
ylabel('PCA2 explained variance='+string(explained(2)))
saveas(gcf,'output_image/NMR/PCA.png')
% t-SNE
rng default
feature_tsne = tsne(zdata);
h_tsne = gscatter(feature_tsne(:,1),feature_tsne(:,2),labels,[],[],10);
title('t-SNE 2D projection of NMR raw data')
xlabel('t-SNE 1') 
ylabel('t-SNE 2')
saveas(gcf,'output_image/NMR/t-SNE.png')

% split training and test
%rng default
%cv = cvpartition(size(data,1),'HoldOut',0.3);
%idx = cv.training;
idx_ = zeros(267,1);
for fileno = 1:size(imgsTrain1.Files,1)
    [filepath,name,ext] = fileparts(imgsTrain1.Files{fileno});
    idx_(str2num(name(3:end))) = 1;
end
idx_ = logical(idx_);
data_train = data(idx_,:);
data_test  = data(~idx_,:);
labels_train = labels(idx_,:);
labels_test = labels(~idx_,:);


%% baseline model
% linear SVM
rng default
Mdlsvm = fitcecoc(normalize(data_train),labels_train,...
                  'FitPosterior',true,'Prior','uniform');
% random forest
tTree = templateTree('surrogate','on');
tEnsemble = templateEnsemble('GentleBoost',100,tTree);
options = statset('UseParallel',true);
rng default
Mdlt = fitcecoc(data_train,labels_train,'FitPosterior',true,...
                'Learners',tEnsemble,'NumBins',10,...
                'Prior','uniform','Options',options);

% test 
%[labels_svm_train,~,~,Posterior_svm_train] = predict(Mdlsvm,normalize(data_train));
[labels_svm_test,~,~,Posterior_svm_test] = predict(Mdlsvm,normalize(data_test));
%[labels_t_train,~,~,Posterior_t_train] = predict(Mdlt,data_train);
[labels_t_test,~,~,Posterior_t_test] = predict(Mdlt,data_test);

%[Xsvm_train,Ysvm_train,Tsvm_train,AUCsvm_train] = perfcurve(labels_train,Posterior_svm_train(:, 2),1);
[Xsvm_test,Ysvm_test,Tsvm_test,AUCsvm_test] = perfcurve(labels_test,Posterior_svm_test(:, 2),1);
%[Xt_train,Yt_train,Tt_train,AUCt_train] = perfcurve(labels_train,Posterior_t_train(:, 2),1);
[Xt_test,Yt_test,Tt_test,AUCt_test] = perfcurve(labels_test,Posterior_t_test(:, 2),1);

cm_svm = confusionmat(labels_test,labels_svm_test);
cm_t = confusionmat(labels_test,labels_t_test);
precision_svm = cm_svm(2,2)/(cm_svm(2,2)+cm_svm(1,2));
recall_svm = cm_svm(2,2)/(cm_svm(2,2)+cm_svm(2,1));
accuracy_svm = (cm_svm(1,1)+cm_svm(2,2))/sum(cm_svm,'all');
precision_t = cm_t(2,2)/(cm_t(2,2)+cm_t(1,2));
recall_t = cm_t(2,2)/(cm_t(2,2)+cm_t(2,1));
accuracy_t = (cm_t(1,1)+cm_t(2,2))/sum(cm_t,'all');

% plot 
subplot(1,2,1)
confusionchart(cm_svm,{'control','IBS'},'Title','Linear SVM classification')
subplot(1,2,2)
plot(Xsvm_test,Ysvm_test)
xlabel('False positive rate')
ylabel('True positive rate')
title('Linear SVM ROC curve')
saveas(gcf,'output_image/NMR/svm.png')

subplot(1,2,1)
confusionchart(cm_t,{'control','IBS'},'Title','Random Forest classification')
subplot(1,2,2)
plot(Xt_test,Yt_test)
xlabel('False positive rate')
ylabel('True positive rate')
title('Random forest ROC curve')
saveas(gcf,'output_image/NMR/rf.png')

Y_as = [accuracy_svm,precision_svm,recall_svm,AUCsvm_test;accuracy_t,precision_t,recall_t,AUCt_test];
X_as = categorical({'Linear SVM','Random Forest'});
HB = bar(X_as, Y_as , 'group');
ylim([0 1])
legend('accuracy','precision','recall','AUC')
a = (1:size(Y_as,1)).';
x = [a-0.3 a-0.1 a+0.1 a+0.3];
for k=1:size(Y_as,1)
    for m = 1:size(Y_as,2)
        text(x(k,m),Y_as(k,m),num2str(Y_as(k,m),'%0.4f'),...
            'HorizontalAlignment','center',...
            'VerticalAlignment','bottom')
    end
end
grid on
grid minor


%% image generation
% with bump wavelet
mkdir(fullfile('data_images/IBS/bump','0'));
mkdir(fullfile('data_images/IBS/bump','1'));
for ii = 1:size(data, 1)
    newx = data(ii, :);
    % data normalisation - sqrt
    % newx = newx-min(newx);
    % newx = sqrt(newx);
    % data normalisation - log
    % newx = newx-min(newx)+0.00001;
    % newx = log(newx);
    % data normalisation - arccosine
    % max_ = max(newx);
    % min_ = min(newx);
    % newx = (2*newx - min_ -max_)/(max_-min_);
    % newx = real(acos(newx));

    %shift left 0.1 ppm
    % noisyregion = newx(1:300);
    % x = randi(size(noisyregion,1),1,200);
    % newintensities = noisyregion(x);
    % newx = [newx(201:end),newintensities];

    [cfs,f] = cwt(newx,'bump',2000,'VoicePerOctave',48);
    im = abs(cfs);
    im = im / max(im(:)) * 255;
    im = uint8(im);
    im = histeq(im);
    % convert the image into rgb
    im = ind2rgb((im),jet(256));
    % im = ind2rgb((im),parula(256));

    % im = im / max(im(:)) * 65535;
    % im = uint16(im);
    % im = histeq(im);
    % im = ind2rgb((im),jet(65536));

    % save the image in jpg file
    imgLoc = fullfile('data_images/IBS/bump',num2str(labels(ii)));
    imFileName = strcat(num2str(labels(ii)),'_',num2str(ii),'.jpg');
    imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
end

% with markov transition field
mkdir(fullfile('data_images/IBS/mtf','0'));
mkdir(fullfile('data_images/IBS/mtf','1'));
poolsize = 5;
edges = [0,120,300,600,1000,1500,3000,5000,10000];
for ii = 1:size(data,1)
    % data pooling to reduce computation - average pooling
    newx = dlarray(data(ii,:),'T');
    newx = maxpool(newx,poolsize,'PoolFormat','T','Stride',poolsize);
    newx = extractdata(newx);
    newsize = size(newx,1);
    % newx = log(newx-min(newx)+0.0001);
    % discretise data into 1900 bins
    disnewx = discretize(newx,edges);
    q = zeros(size(edges,2)-1, size(edges,2)-1);
    for jj = 1:size(newx,1)-1
        q(disnewx(jj),disnewx(jj+1)) = q(disnewx(jj),disnewx(jj+1))+1;
    end
    q = q./sum(q,2);
    mtf = zeros(newsize, newsize);
    for i = 1:newsize
        for j = 1:newsize
            mtf(i,j) = q(disnewx(i),disnewx(j));
        end
    end
    im = mtf * 255;
    im = uint8(im);
    im = histeq(im);
    % convert the image into rgb
    im = ind2rgb((im),jet(256));
    % save the image in jpg file
    imgLoc = fullfile('data_images/IBS/mtf',num2str(labels(ii)));
    imFileName = strcat(num2str(labels(ii)),'_',num2str(ii),'.jpg');
    imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
end

% with Gramian angular field
mkdir(fullfile('data_images/IBS/gasfavg5','0'));
mkdir(fullfile('data_images/IBS/gasfavg5','1'));
mkdir(fullfile('data_images/IBS/gadfavg5','0'));
mkdir(fullfile('data_images/IBS/gadfavg5','1'));
poolsize = 5;
for ii = 1:size(IBS267wnoise,1)-1
    % data pooling to reduce computation - average pooling
    newx = dlarray(IBS267wnoise(ii+1,2:end),'T');
    newx = avgpool(newx,poolsize,'PoolFormat','T','Stride',poolsize);
    newx = extractdata(newx);
    newsize = size(newx,1);

    max_ = max(newx);
    min_ = min(newx);
    newx = (2*newx - min_ -max_)/(max_-min_);
    newx = real(acos(newx));
    newm = repmat(newx,1,newsize);
    newmt = transpose(newm);
    % Gramian angular summation field
    gasf = cos(newm+newmt);
    % Gramian angular difference field
    gadf = sin(newm-newmt);

    % image generation with gasf and gadf
    gasf = (gasf+1)/2*255;
    gasf = uint8(gasf);
    gasf = histeq(gasf);
    gadf = (gadf+1)/2*255;
    gadf = uint8(gadf);
    gadf = histeq(gadf);

    % convert the image into rgb
    gasf = ind2rgb((gasf),jet(256));
    gadf = ind2rgb((gadf),jet(256));
    
    % save the image in jpg file
    imgLocs = fullfile('data_images/IBS/gasfavg5',num2str(IBS267wnoise(ii+1,1)));
    imgLocd = fullfile('data_images/IBS/gadfavg5',num2str(IBS267wnoise(ii+1,1)));
    imFileName = strcat(num2str(IBS267wnoise(ii+1,1)),'_',num2str(ii),'.jpg');
    imwrite(imresize(gasf,[224 224]),fullfile(imgLocs,imFileName));
    imwrite(imresize(gadf,[224 224]),fullfile(imgLocd,imFileName));
end


%% CNN training
allImages = imageDatastore(fullfile('data_images/IBS/morse'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
rng default
% allImages = balanceImageDatastore(allImages,303);
[imgsTrain1,imgsTest1] = splitEachLabel(allImages,0.7,'randomized');
numClasses = numel(categories(imgsTrain1.Labels));

% googlenet
net1 = googlenet;
lgraph1 = layerGraph(net1);
newDropoutLayer1 = dropoutLayer(0.6,'Name','new_Dropout');
lgraph1 = replaceLayer(lgraph1,'pool5-drop_7x7_s1',newDropoutLayer1);
newConnectedLayer1 = fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
lgraph1 = replaceLayer(lgraph1,'loss3-classifier',newConnectedLayer1);
newClassLayer1 = classificationLayer('Name','new_classoutput');
lgraph1 = replaceLayer(lgraph1,'output',newClassLayer1);

% set training options and train
options = trainingOptions('adam',...
'MiniBatchSize',64,...
'MaxEpochs',15,...
'InitialLearnRate',2e-5,...
'ValidationData',imgsTest1,...
'ValidationFrequency',10);

rng default
trainedGN1 = trainNetwork(imgsTrain1,lgraph1,options);
[YPredtrain1,probstrain1] = classify(trainedGN1,imgsTrain1);
% Evaluate GoogLeNet accuracy
[YPred1,probs1] = classify(trainedGN1,imgsTest1);
[X1,Y1,T1,AUC1] = perfcurve(imgsTest1.Labels,probs1(:, 2),1);
disp(['AUC of model 1 is : ',num2str(AUC1)])
cm_1 = confusionmat(imgsTest1.Labels,YPred1);
precision_1 = cm_1(2,2)/(cm_1(2,2)+cm_1(1,2));
recall_1 = cm_1(2,2)/(cm_1(2,2)+cm_1(2,1));
accuracy_1 = (cm_1(1,1)+cm_1(2,2))/sum(cm_1,'all');

% permutation test on AUC label
test_labels = imgsTest1.Labels;
[~,~,~,AUC1] = perfcurve(test_labels,probs1(:, 2),1);
auc_permutated = zeros(10000,1);
for ii = 1:10000
    [~,~,~,AUC_] = perfcurve(test_labels(randperm(length(test_labels))),probs1(:, 2),1);
    auc_permutated(ii) = AUC_;
end
p_valeu = (sum(auc_permutated>=AUC1)+1)/(10000+1);

% cross validation cnn
datastore = imageDatastore(fullfile('morse'),'IncludeSubfolders',true,'LabelSource','foldernames');
[imds1,imds2,imds3,imds4,imds5] = splitEachLabel(datastore,0.2,0.2,0.2,0.2,'randomize');
imds = {imds1.Files,imds2.Files,imds3.Files,imds4.Files,imds5.Files};
AUC_cv = 0;
accuracy_cv = 0;
precision_cv = 0;
recall_cv = 0;
for ii = 1:5
    imgsTrain = imageDatastore(cat(1, imds{[1:ii-1 ii+1:end]}),'IncludeSubfolders',true,'LabelSource','foldernames');
    imgsTest = imageDatastore(imds{ii},'IncludeSubfolders',true,'LabelSource','foldernames');
    numClasses = numel(categories(imgsTrain.Labels));
    % googlenet
    net = googlenet;
    lgraph = layerGraph(net);
    newDropoutLayer = dropoutLayer(0.6,'Name','new_Dropout');
    lgraph = replaceLayer(lgraph,'pool5-drop_7x7_s1',newDropoutLayer);
    newConnectedLayer = fullyConnectedLayer(numClasses,'Name','new_fc',...
        'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
    lgraph = replaceLayer(lgraph,'loss3-classifier',newConnectedLayer);
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'output',newClassLayer);
    
    % set training options and train
    options = trainingOptions('adam',...
    'MiniBatchSize',64,...
    'MaxEpochs',15,...
    'InitialLearnRate',2e-5,...
    'ValidationData',imgsTest,...
    'ValidationFrequency',10);
    
    rng default
    trainedGN = trainNetwork(imgsTrain,lgraph,options);
    [YPredtrain,probstrain] = classify(trainedGN,imgsTrain);
    % Evaluate GoogLeNet accuracy
    [YPred,probs] = classify(trainedGN,imgsTest);
    [X,Y,~,AUC_cv_] = perfcurve(imgsTest.Labels,probs(:, 1),1);
    AUC_cv = AUC_cv+AUC_cv_/5;
    cm = confusionmat(imgsTest.Labels,YPred);
    precision_cv = precision_cv+cm(2,2)/(cm(2,2)+cm(1,2))/5;
    recall_cv = recall_cv+cm(2,2)/(cm(2,2)+cm(2,1))/5;
    accuracy_cv = accuracy_cv+(cm(1,1)+cm(2,2))/sum(cm,'all')/5;
end

% second model
allImages = imageDatastore(fullfile('bump'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
rng default
[imgsTrain2,imgsTest2] = splitEachLabel(allImages,0.7,'randomized');
net2 = googlenet;
lgraph2 = layerGraph(net2);
newDropoutLayer2 = dropoutLayer(0.6,'Name','new_Dropout');
lgraph2 = replaceLayer(lgraph2,'pool5-drop_7x7_s1',newDropoutLayer2);
newConnectedLayer2 = fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
lgraph2 = replaceLayer(lgraph2,'loss3-classifier',newConnectedLayer2);
newClassLayer2 = classificationLayer('Name','new_classoutput');
lgraph2 = replaceLayer(lgraph2,'output',newClassLayer2);
rng default
trainedGN2 = trainNetwork(imgsTrain2,lgraph2,options);
[YPredtrain2,probstrain2] = classify(trainedGN2,imgsTrain2);
[YPred2,probs2] = classify(trainedGN2,imgsTest2);
[X2,Y2,T2,AUC2] = perfcurve(imgsTest2.Labels,probs2(:, 2),1);
disp(['AUC of model 2 is : ',num2str(AUC2)])
cm_2 = confusionmat(imgsTest2.Labels,YPred2);
precision_2 = cm_2(2,2)/(cm_2(2,2)+cm_2(1,2));
recall_2 = cm_2(2,2)/(cm_2(2,2)+cm_2(2,1));
accuracy_2 = (cm_2(1,1)+cm_2(2,2))/sum(cm_2,'all');

allImages = imageDatastore(fullfile('bump'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
rng default
[imgsTrain3,imgsTest3] = splitEachLabel(allImages,0.7,'randomized');
net3 = googlenet;
lgraph3 = layerGraph(net3);
newDropoutLayer3 = dropoutLayer(0.6,'Name','new_Dropout');
lgraph3 = replaceLayer(lgraph3,'pool5-drop_7x7_s1',newDropoutLayer3);
newConnectedLayer3 = fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
lgraph3 = replaceLayer(lgraph3,'loss3-classifier',newConnectedLayer3);
newClassLayer3 = classificationLayer('Name','new_classoutput');
lgraph3 = replaceLayer(lgraph3,'output',newClassLayer3);
% resnet50
rnet = resnet50;
rlgraph = layerGraph(rnet);
rnewDropoutLayer = dropoutLayer(0.6,'Name','new_Dropout');
rlgraph = addLayers(rlgraph,rnewDropoutLayer);
rlgraph = disconnectLayers(rlgraph,'avg_pool','fc1000');
rlgraph = connectLayers(rlgraph,'avg_pool','new_Dropout');
rlgraph = connectLayers(rlgraph,'new_Dropout','fc1000');
rnewConnectedLayer = fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
rlgraph = replaceLayer(rlgraph,'fc1000',rnewConnectedLayer);
rnewClassLayer = classificationLayer('Name','new_classoutput');
rlgraph = replaceLayer(rlgraph,'ClassificationLayer_fc1000',rnewClassLayer);
rng default
trainedGN3 = trainNetwork(imgsTrain3,lgraph3,options);
[YPredtrain3,probstrain3] = classify(trainedGN3,imgsTrain3);
[YPred3,probs3] = classify(trainedGN3,imgsTest3);
[X3,Y3,T3,AUC3] = perfcurve(imgsTest3.Labels,probs3(:, 2),1);
disp(['AUC of model 3 is : ',num2str(AUC3)])
cm_3 = confusionmat(imgsTest3.Labels,YPred3);
precision_3 = cm_3(2,2)/(cm_3(2,2)+cm_3(1,2));
recall_3 = cm_3(2,2)/(cm_3(2,2)+cm_3(2,1));
accuracy_3 = (cm_3(1,1)+cm_3(2,2))/sum(cm_3,'all');

allImages = imageDatastore(fullfile('mtf_arccos'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
rng default
[imgsTrain4,imgsTest4] = splitEachLabel(allImages,0.7,'randomized');
net4 = googlenet;
lgraph4 = layerGraph(net4);
newDropoutLayer4 = dropoutLayer(0.6,'Name','new_Dropout');
lgraph4 = replaceLayer(lgraph4,'pool5-drop_7x7_s1',newDropoutLayer4);
newConnectedLayer4 = fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
lgraph4 = replaceLayer(lgraph4,'loss3-classifier',newConnectedLayer4);
newClassLayer4 = classificationLayer('Name','new_classoutput');
lgraph4 = replaceLayer(lgraph4,'output',newClassLayer4);
rng default
trainedGN4 = trainNetwork(imgsTrain4,lgraph4,options);
[YPredtrain4,probstrain4] = classify(trainedGN4,imgsTrain4);
[YPred4,probs4] = classify(trainedGN4,imgsTest4);
[X4,Y4,T4,AUC4] = perfcurve(imgsTest4.Labels,probs4(:, 2),1);
disp(['AUC of model 4 is : ',num2str(AUC4)])
cm_4 = confusionmat(imgsTest4.Labels,YPred4);
precision_4 = cm_4(2,2)/(cm_4(2,2)+cm_4(1,2));
recall_4 = cm_4(2,2)/(cm_4(2,2)+cm_4(2,1));
accuracy_4 = (cm_4(1,1)+cm_4(2,2))/sum(cm_4,'all');

allImages = imageDatastore(fullfile('gasfmax10'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
rng default
[imgsTrain5,imgsTest5] = splitEachLabel(allImages,0.7,'randomized');
net5 = googlenet;
lgraph5 = layerGraph(net5);
newDropoutLayer5 = dropoutLayer(0.6,'Name','new_Dropout');
lgraph5 = replaceLayer(lgraph5,'pool5-drop_7x7_s1',newDropoutLayer5);
newConnectedLayer5 = fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
lgraph5 = replaceLayer(lgraph5,'loss3-classifier',newConnectedLayer5);
newClassLayer5 = classificationLayer('Name','new_classoutput');
lgraph5 = replaceLayer(lgraph5,'output',newClassLayer5);
rng default
trainedGN5 = trainNetwork(imgsTrain5,lgraph5,options);
[YPredtrain5,probstrain5] = classify(trainedGN5,imgsTrain5);
[YPred5,probs5] = classify(trainedGN5,imgsTest5);
[X5,Y5,T5,AUC5] = perfcurve(imgsTest5.Labels,probs5(:, 2),1);
disp(['AUC of model 5 is : ',num2str(AUC5)])
cm_5 = confusionmat(imgsTest5.Labels,YPred5);
precision_5 = cm_5(2,2)/(cm_5(2,2)+cm_5(1,2));
recall_5 = cm_5(2,2)/(cm_5(2,2)+cm_5(2,1));
accuracy_5 = (cm_5(1,1)+cm_5(2,2))/sum(cm_5,'all');

allImages = imageDatastore(fullfile('gadfmax5'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
rng default
[imgsTrain6,imgsTest6] = splitEachLabel(allImages,0.7,'randomized');
net6 = googlenet;
lgraph6 = layerGraph(net6);
newDropoutLayer6 = dropoutLayer(0.6,'Name','new_Dropout');
lgraph6 = replaceLayer(lgraph6,'pool5-drop_7x7_s1',newDropoutLayer6);
newConnectedLayer6 = fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
lgraph6 = replaceLayer(lgraph6,'loss3-classifier',newConnectedLayer6);
newClassLayer6 = classificationLayer('Name','new_classoutput');
lgraph6 = replaceLayer(lgraph6,'output',newClassLayer6);
rng default
trainedGN6 = trainNetwork(imgsTrain6,lgraph6,options);
[YPredtrain6,probstrain6] = classify(trainedGN6,imgsTrain6);
[YPred6,probs6] = classify(trainedGN6,imgsTest6);
[X6,Y6,T6,AUC6] = perfcurve(imgsTest6.Labels,probs6(:, 2),1);
disp(['AUC of model 6 is : ',num2str(AUC6)])
cm_6 = confusionmat(imgsTest6.Labels,YPred6);
precision_6 = cm_6(2,2)/(cm_6(2,2)+cm_6(1,2));
recall_6 = cm_6(2,2)/(cm_6(2,2)+cm_6(2,1));
accuracy_6 = (cm_6(1,1)+cm_6(2,2))/sum(cm_6,'all');

% plot six base models
Y_as = [accuracy_1,precision_1,recall_1,AUC1;
        accuracy_2,precision_2,recall_2,AUC2;
        accuracy_3,precision_3,recall_3,AUC3;
        accuracy_4,precision_4,recall_4,AUC4;
        accuracy_5,precision_5,recall_5,AUC5;
        accuracy_6,precision_6,recall_6,AUC6];
catStrArray = {'Morse','Morlet','Bump','MTF','GASF','GADF'};
X_as = categorical(catStrArray);       
X_as = reordercats(X_as,catStrArray);
HB = bar(X_as, Y_as , 'group');
ylim([0 1])
legend('accuracy','precision','recall','AUC')
a = (1:size(Y_as,1)).';
x = [a-0.3 a-0.1 a+0.1 a+0.3];
for k=1:size(Y_as,1)
    for m = 1:size(Y_as,2)
        text(x(k,m),Y_as(k,m),num2str(Y_as(k,m),'%0.2f'),...
            'HorizontalAlignment','center',...
            'VerticalAlignment','bottom')
    end
end
grid on
grid minor

% stacking ensemble
traindata = [probstrain1(:, 2),probstrain2(:, 2),probstrain3(:, 2),probstrain4(:, 2),probstrain5(:, 2),probstrain6(:, 2)];
test = [probs1(:, 2),probs2(:, 2),probs3(:, 2),probs4(:, 2),probs5(:, 2),probs6(:, 2)];
layers = [
    featureInputLayer(6,'Name','input')
    fullyConnectedLayer(20, 'Name','fc1')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2, 'Name','fc2')
    softmaxLayer('Name','sm')
    classificationLayer('Name','classification')];
options = trainingOptions('adam', ...
    'MiniBatchSize',10, ...
    'Shuffle','every-epoch', ...
    'Verbose',false);
rng default
trainednet = trainNetwork(traindata,imgsTrain1.Labels,layers,options);
[YPred,probs] = classify(trainednet,test,'MiniBatchSize',10);
[X,Y,~,AUC] = perfcurve(imgsTest1.Labels,probs(:, 2),1);

cm = confusionmat(imgsTest1.Labels,YPred);
precision = cm(2,2)/(cm(2,2)+cm(1,2));
recall = cm(2,2)/(cm(2,2)+cm(2,1));
accuracy = (cm(1,1)+cm(2,2))/sum(cm,'all');

% linear svm ensemble
Mdllc = fitcecoc(double(traindata),imgsTrain1.Labels,...
                  'FitPosterior',true,'Prior','uniform');
[YPred_,~,~,probs_] = predict(Mdllc,double(test));
[X_,Y_,~,AUC_] = perfcurve(imgsTest1.Labels,probs_(:, 2),1);
cm_ = confusionmat(imgsTest1.Labels,YPred_);
precision_ = cm_(2,2)/(cm_(2,2)+cm_(1,2));
recall_ = cm_(2,2)/(cm_(2,2)+cm_(2,1));
accuracy_ = (cm_(1,1)+cm_(2,2))/sum(cm_,'all');

% plot ensemble models
Y_as = [accuracy,precision,recall,AUC;
        accuracy_,precision_,recall_,AUC_];
catStrArray = {'NN','SVM'};
X_as = categorical(catStrArray);       
X_as = reordercats(X_as,catStrArray);
HB = bar(X_as, Y_as , 'group');
ylim([0 1])
legend('accuracy','precision','recall','AUC')
a = (1:size(Y_as,1)).';
x = [a-0.3 a-0.1 a+0.1 a+0.3];
for k=1:size(Y_as,1)
    for m = 1:size(Y_as,2)
        text(x(k,m),Y_as(k,m),num2str(Y_as(k,m),'%0.2f'),...
            'HorizontalAlignment','center',...
            'VerticalAlignment','bottom')
    end
end
grid on
grid minor

%% signal shift
% baseline models
noisyregion = data(:,1:300);
x = randi(size(noisyregion,2),1,200);
newintensities = noisyregion(:,x);
data_shift = [newintensities,data(:,1:end-200)];

[labels_svm_shift,~,~,Posterior_svm_shift] = predict(Mdlsvm,normalize(data_shift));
[labels_t_shift,~,~,Posterior_t_shift] = predict(Mdlt,data_shift);
[Xsvm_shift,Ysvm_shift,Tsvm_shift,AUCsvm_shift] = perfcurve(labels,Posterior_svm_shift(:, 2),1);
[Xt_shift,Yt_shift,Tt_shift,AUCt_shift] = perfcurve(labels,Posterior_t_shift(:, 2),1);
cm_svm_shift = confusionmat(labels,labels_svm_shift);
cm_t_shift = confusionmat(labels,labels_t_shift);
precision_svm_shift = cm_svm_shift(2,2)/(cm_svm_shift(2,2)+cm_svm_shift(1,2));
recall_svm_shift = cm_svm_shift(2,2)/(cm_svm_shift(2,2)+cm_svm_shift(2,1));
accuracy_svm_shift = (cm_svm_shift(1,1)+cm_svm_shift(2,2))/sum(cm_svm_shift,'all');
precision_t_shift = cm_t_shift(2,2)/(cm_t_shift(2,2)+cm_t_shift(1,2));
recall_t_shift = cm_t_shift(2,2)/(cm_t_shift(2,2)+cm_t_shift(2,1));
accuracy_t_shift = (cm_t_shift(1,1)+cm_t_shift(2,2))/sum(cm_t_shift,'all');

% plot
subplot(1,2,1)
confusionchart(cm_svm_shift,{'control','IBS'},'Title','linear SVM classification')
subplot(1,2,2)
plot(Xsvm_shift,Ysvm_shift)
xlabel('False positive rate')
ylabel('True positive rate')
title('linear SVM ROC curve')

subplot(1,2,1)
confusionchart(cm_t_shift,{'control','IBS'},'Title','Random forest classification')
subplot(1,2,2)
plot(Xt_shift,Yt_shift)
xlabel('False positive rate')
ylabel('True positive rate')
title('Random forest ROC curve')


% CNN model test on shifted data
imgsTest1_ = imageDatastore(fullfile('morse'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
[YPred1_,probs1_] = classify(trainedGN1_,imgsTest1_);
[X1_,Y1_,T1_,AUC1_] = perfcurve(imgsTest1_.Labels,probs1_(:, 2),1);
disp(['AUC of model 1 is : ',num2str(AUC1_)])
cm_1_ = confusionmat(imgsTest1_.Labels,YPred1_);
precision_1_ = cm_1_(2,2)/(cm_1_(2,2)+cm_1_(1,2));
recall_1_ = cm_1_(2,2)/(cm_1_(2,2)+cm_1_(2,1));
accuracy_1_ = (cm_1_(1,1)+cm_1_(2,2))/sum(cm_1_,'all');

imgsTest2_ = imageDatastore(fullfile('morlet_arccos'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
[YPred2_,probs2_] = classify(trainedGN2_,imgsTest2_);
[X2_,Y2_,T2_,AUC2_] = perfcurve(imgsTest2_.Labels,probs2_(:, 2),1);
disp(['AUC of model 2 is : ',num2str(AUC2_)])
cm_2_ = confusionmat(imgsTest2_.Labels,YPred2_);
precision_2_ = cm_2_(2,2)/(cm_2_(2,2)+cm_2_(1,2));
recall_2_ = cm_2_(2,2)/(cm_2_(2,2)+cm_2_(2,1));
accuracy_2_ = (cm_2_(1,1)+cm_2_(2,2))/sum(cm_2_,'all');

imgsTest3_ = imageDatastore(fullfile('bump'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
[YPred3_,probs3_] = classify(trainedGN3_,imgsTest3_);
[X3_,Y3_,T3_,AUC3_] = perfcurve(imgsTest3_.Labels,probs3_(:, 2),1);
disp(['AUC of model 3 is : ',num2str(AUC3_)])
cm_3_ = confusionmat(imgsTest3_.Labels,YPred3_);
precision_3_ = cm_3_(2,2)/(cm_3_(2,2)+cm_3_(1,2));
recall_3_ = cm_3_(2,2)/(cm_3_(2,2)+cm_3_(2,1));
accuracy_3_ = (cm_3_(1,1)+cm_3_(2,2))/sum(cm_3_,'all');

imgsTest4_ = imageDatastore(fullfile('mtf_arccos'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
[YPred4_,probs4_] = classify(trainedGN4_,imgsTest4_);
[X4_,Y4_,T4_,AUC4_] = perfcurve(imgsTest4_.Labels,probs4_(:, 2),1);
disp(['AUC of model 4 is : ',num2str(AUC4_)])
cm_4_ = confusionmat(imgsTest4_.Labels,YPred4_);
precision_4_ = cm_4_(2,2)/(cm_4_(2,2)+cm_4_(1,2));
recall_4_ = cm_4_(2,2)/(cm_4_(2,2)+cm_4_(2,1));
accuracy_4_ = (cm_4_(1,1)+cm_4_(2,2))/sum(cm_4_,'all');

imgsTest5_ = imageDatastore(fullfile('gasfmax10'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
[YPred5_,probs5_] = classify(trainedGN5_,imgsTest5_);
[X5_,Y5_,T5_,AUC5_] = perfcurve(imgsTest5_.Labels,probs5_(:, 2),1);
disp(['AUC of model 5 is : ',num2str(AUC5_)])
cm_5_ = confusionmat(imgsTest5_.Labels,YPred5_);
precision_5_ = cm_5_(2,2)/(cm_5_(2,2)+cm_5_(1,2));
recall_5_ = cm_5_(2,2)/(cm_5_(2,2)+cm_5_(2,1));
accuracy_5_ = (cm_5_(1,1)+cm_5_(2,2))/sum(cm_5_,'all');

imgsTest6_ = imageDatastore(fullfile('gadfmax5'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
[YPred6_,probs6_] = classify(trainedGN6_,imgsTest6_);
[X6_,Y6_,T6_,AUC6_] = perfcurve(imgsTest6_.Labels,probs6_(:, 2),1);
disp(['AUC of model 6 is : ',num2str(AUC6_)])
cm_6_ = confusionmat(imgsTest6_.Labels,YPred6_);
precision_6_ = cm_6_(2,2)/(cm_6_(2,2)+cm_6_(1,2));
recall_6_ = cm_6_(2,2)/(cm_6_(2,2)+cm_6_(2,1));
accuracy_6_ = (cm_6_(1,1)+cm_6_(2,2))/sum(cm_6_,'all');

% stacking ensemble
test_ = [probs1_(:, 2),probs2_(:, 2),probs3_(:, 2),probs4_(:, 2),probs5_(:, 2),probs6_(:, 2)];
[YPred,probs] = classify(trainednet,test_,'MiniBatchSize',10);
[X,Y,T,AUC] = perfcurve(imgsTest1_.Labels,probs(:, 2),1);
cm = confusionmat(imgsTest1_.Labels,YPred);
precision = cm(2,2)/(cm(2,2)+cm(1,2));
recall = cm(2,2)/(cm(2,2)+cm(2,1));
accuracy = (cm(1,1)+cm(2,2))/sum(cm,'all');
% plot
subplot(1,2,1)
confusionchart(cm,{'control','IBS'},'Title','stacking CNN classification')
subplot(1,2,2)
plot(X,Y)
xlabel('False positive rate')
ylabel('True positive rate')
title('stacking CNN ROC curve')

% linear svm
[YPred_,~,~,probs_] = predict(Mdllc,double(test_));
[X_,Y_,T_,AUC_] = perfcurve(imgsTest1_.Labels,probs_(:, 2),1);
cm_ = confusionmat(imgsTest1_.Labels,YPred_);
precision_ = cm_(2,2)/(cm_(2,2)+cm_(1,2));
recall_ = cm_(2,2)/(cm_(2,2)+cm_(2,1));
accuracy_ = (cm_(1,1)+cm_(2,2))/sum(cm_,'all');
% plot
subplot(1,2,1)
confusionchart(cm_,{'control','IBS'},'Title','linear ensemble')
subplot(1,2,2)
plot(X_,Y_)
xlabel('False positive rate')
ylabel('True positive rate')
title('linear ensemble CNN ROC curve')

% plot ensemble models
Y_as = [accuracy,precision,recall,AUC;
        accuracy_,precision_,recall_,AUC_];
catStrArray = {'NN','SVM'};
X_as = categorical(catStrArray);       
X_as = reordercats(X_as,catStrArray);
HB = bar(X_as, Y_as , 'group');
ylim([0 1])
legend('accuracy','precision','recall','AUC')
a = (1:size(Y_as,1)).';
x = [a-0.3 a-0.1 a+0.1 a+0.3];
for k=1:size(Y_as,1)
    for m = 1:size(Y_as,2)
        text(x(k,m),Y_as(k,m),num2str(Y_as(k,m),'%0.2f'),...
            'HorizontalAlignment','center',...
            'VerticalAlignment','bottom')
    end
end
grid on
grid minor


%% extract informative regions
img = imread("data_images/IBS/bump/0/0_1.jpg");
map_gcam = gradCAM(gtrainedGN,img,1);
map_gb = gradientMap(gtrainedGN,img,'GuidedBackprop');
map_ge = gradientMap(gtrainedGN,img,'GradientExplanation'); 
map_zf = gradientMap(gtrainedGN,img,'ZeilerFergus');
map_oc = occlusionSensitivity(gtrainedGN,img,1);

map = zeros(224,224);
filePattern = fullfile('data_images/IBS/bump/0/*.jpg');
theFiles = dir(filePattern);
ff = zeros(length(theFiles),224);
for k = 1:length(theFiles)
    FileName = fullfile('data_images/IBS/bump/0',theFiles(k).name);
    img = imread(FileName);
    map_gcam = gradCAM(gtrainedGN,img,1);
    map_gb = gradientMap(gtrainedGN,img,'GuidedBackprop');
    map = map+map_gcam.*double(map_gb);
    ff(k,:) = sum(map>30,1);
end
map = map/k;
imagesc(map);
newmz = interp1(1:19000,ppm,linspace(1,19000,224),'nearest');
plot(newmz,sum(ff,1))

%reconstruction
img = imread("/Users/cl6915/Documents/MATLAB/data_images/IBS/morse/0/0_1.jpg");
[cfs,~] = cwt(data(1,:),'morse',2000,'VoicePerOctave',48);
%GRAD-CAM
map_gcam = gradCAM(trainedGN1,img,2);
map_gcam_shallow = gradCAM(net,img,YPred,'FeatureLayer',"conv2-relu_3x3");
map_ = map_gcam>0.3;
%occlusion map
%map_oc = occlusionSensitivity(trainedGN1,img,categorical(1));
%map_ = (map_oc>0).*map_oc*5;
%back propagation
%map_gb = gradientMap(trainedGN1,img,'GuidedBackprop');
%map_ = map_gb>5;
%GRAD-CAM & back propagation
%map = double(map_gb).*map_gcam;
%map_ = map>2;

map_ = imresize(map_,[542 19000]);
%original cwt
wmap = map_.*cfs;
%absolute cwt
%wmap = map_.*abs(cfs);
xrec = icwt(wmap,'morse',VoicesPerOctave=48);
%factor = sum(xrec.*data(2,2:end))/sum(xrec.^2);
plot(ppm,data(2,2:end));
set(gca, 'XDir','reverse')
hold on
plot(ppm,xrec);
hold off
legend('original','reconstructed','Location','northwest')
%plot(ppm,xrec.*(xrec>4000))


filePattern = fullfile('/Users/cl6915/Documents/MATLAB/data_images/IBS/morse/0/*.jpg');
theFiles = dir(filePattern);
map_0 = zeros(224,224);
for k = 1:length(theFiles)
    img = imread(fullfile(theFiles(k).folder,theFiles(k).name));
    map_oc = occlusionSensitivity(trainedGN1,img,categorical(0));
    map_gb = gradientMap(trainedGN1,img,'GuidedBackprop');
    map_ = map_oc>0;
    map_0 = map_0+double(map_gb).*map_.*map_oc;
end
map_0 = map_0/k;
plot(linspace(0.5005,9.9996,224),sum(map_0,1))


%% simulated NMR data
simulated_class = readmatrix('data/IBS/simulated_groups.xlsx','Sheet','simulated_class');
simulated_class = simulated_class(:,2);
pred_simulated = zeros(size(simulated_class,1));
region = readmatrix('data/IBS/simulated_groups.xlsx','Sheet','important_variables');
region_1_1 = region(145:188,1)+400;
region_1_2 = region(189:234,1)+400;
region_2_1 = region(1:61,1);
region_2_2 = region(62:144,1);

for ii = 1:267
    movefile(ii+".jpg",simulated_class(ii)+"/"+ii+".jpg");
end

allImages = imageDatastore(fullfile('morse_simulated'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
rng default
[imgsTrain,imgsTest] = splitEachLabel(allImages,0.7,'randomized');
numClasses = numel(categories(imgsTrain.Labels));
gnet = googlenet;
glgraph = layerGraph(gnet);
newDropoutLayer1 = dropoutLayer(0.6,'Name','new_Dropout');
glgraph = replaceLayer(glgraph,'pool5-drop_7x7_s1',newDropoutLayer1);
gnewConnectedLayer = fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
glgraph = replaceLayer(glgraph,'loss3-classifier',gnewConnectedLayer);
gnewClassLayer = classificationLayer('Name','new_classoutput');
glgraph = replaceLayer(glgraph,'output',gnewClassLayer);
options = trainingOptions('adam',...
'MiniBatchSize',64,...
'MaxEpochs',15,...
'InitialLearnRate',2e-5);
rng default
gtrainedGN = trainNetwork(imgsTrain,glgraph,options);
%[YPredgtrain,probsgtrain] = classify(gtrainedGN,imgsTrain);
[YPredgtest,probsgtest] = classify(gtrainedGN,imgsTest);
%[Xgtrain1,Ygtrain1,Tgtrain1,AUCgtrain1] = perfcurve(imgsTrain.Labels,probsgtrain(:, 1),1);
[Xgtest1,Ygtest1,~,AUCgtest1] = perfcurve(imgsTest.Labels,probsgtest(:, 1),1);
disp(['test AUC is : ',num2str(AUCgtest1)])
cm = confusionmat(imgsTest.Labels,YPredgtest);
precision = cm(2,2)/(cm(2,2)+cm(1,2));
recall = cm(2,2)/(cm(2,2)+cm(2,1));
accuracy = (cm(1,1)+cm(2,2))/sum(cm,'all');

for ii = 1:267
    [cfs,~] = cwt(data(ii,:),'morse',2000,'VoicePerOctave',48);
    img_name = "data_images/IBS/morse_simulated/"+simulated_class(ii)+"/"+ii+".jpg";
    img_wavelet = imread(img_name);
    map_gcam = gradCAM(gtrainedGN,img_wavelet,simulated_class(ii));
    map_gb = gradientMap(gtrainedGN,img_wavelet,'GuidedBackprop');

    %grad-cam
    map_ = map_gcam>0.5;
    map_ = imresize(map_,[542 19000]);
    wmap = map_.*cfs;
    xrec = icwt(wmap,'morse',VoicesPerOctave=48);
    xrec_1_1 = xrec(region_1_1);
    spectrum_1_1 = data(ii,region_1_1);
    xrec_1_1 = xrec_1_1/mean(spectrum_1_1);
    spectrum_1_1 = spectrum_1_1/mean(spectrum_1_1);
    xrec_1_2 = xrec(region_1_2);
    spectrum_1_2 = data(ii,region_1_2);
    xrec_1_2 = xrec_1_2/mean(spectrum_1_2);
    spectrum_1_2 = spectrum_1_2/mean(spectrum_1_2);
    xrec_2_1 = xrec(region_2_1);
    spectrum_2_1 = data(ii,region_2_1);
    xrec_2_1 = xrec_2_1/mean(spectrum_2_1);
    spectrum_2_1 = spectrum_2_1/mean(spectrum_2_1);
    xrec_2_2 = xrec(region_2_2);
    spectrum_2_2 = data(ii,region_2_2);
    xrec_2_2 = xrec_2_2/mean(spectrum_2_2);
    spectrum_2_2 = spectrum_2_2/mean(spectrum_2_2);
    
    mser_1 = mean([(xrec_1_1-spectrum_1_1).^2,(xrec_1_2-spectrum_1_2).^2]);
    mser_2 = mean([(xrec_2_1-spectrum_2_1).^2,(xrec_2_2-spectrum_2_2).^2]);

    %gradient backpropagation
    %map_ = map_gb>5;
    %map_ = imresize(map_,[542 19000]);
    %wmap = map_.*cfs;
    %xrec = icwt(wmap,'morse',VoicesPerOctave=48);
    %grad-cam and gradient backpropagation
    %map_ = double(map_gb).*map_gcam;
    %map_ = map_>2;
    %map_ = imresize(map_,[542 19000]);
    %wmap = map_.*cfs;
    %xrec = icwt(wmap,'morse',VoicesPerOctave=48);

    pred_simulated(ii) = (mser_1>mser_2)+1;
end


%% MSI dataset
data = h5read('mouse_bladder.h5','/features',[1,1],[120000,10000]);
data = reshape(data, 2, 60000, 10000);
data = sum(data, 1) * 0.5;
data = reshape(data,60000,10000);
data = transpose(data);
data_1 = h5read('mouse_bladder.h5','/features',[1,10001],[120000,10000]);
data_1 = reshape(data_1, 2, 60000, 10000);
data_1 = sum(data_1, 1) * 0.5;
data_1 = reshape(data_1,60000,10000);
data_1 = transpose(data_1);
data = [data;data_1];
data_1 = h5read('mouse_bladder.h5','/features',[1,20001],[120000,14840]);
data_1 = reshape(data_1, 2, 60000, 14840);
data_1 = sum(data_1, 1) * 0.5;
data_1 = reshape(data_1,60000,14840);
data_1 = transpose(data_1);
data = [data;data_1];
clearvars data_1
%data_compress = reshape(sum(reshape(data,34840,10,6000),2)*0.1,34840,6000);
labels = h5read('mouse_bladder.h5','/labels');
labels = reshape(labels,34840,1);
%data = data(~(labels==0),:);
xy_coords = h5read('mouse_bladder.h5','/xy_coords');
xy_coords = transpose(xy_coords);
%xy_coords = xy_coords(~(labels==0),:);
%labels = labels(~(labels==0),:);
%masses = h5read('mouse_bladder.h5','/masses');
%masses = reshape(sum(reshape(masses,2,60000),1)*0.5,60000,1);
data = data-min(data,[],2);

% PCA on raw data
data_compress = data_compress(~(labels==0),:);
labels_0 = labels(~(labels==0),:);
[~,feature_pca,~] = pca(data_compress);
h_pca = gscatter(feature_pca(:,1),feature_pca(:,2),labels_0,[],[],10);
set(h_pca(1), 'Marker','o', 'MarkerSize',5, 'MarkerEdgeColor','none', 'MarkerFaceColor','b');
set(h_pca(2), 'Marker','o', 'MarkerSize',5, 'MarkerEdgeColor','none', 'MarkerFaceColor','r');
set(h_pca(3), 'Marker','o', 'MarkerSize',5, 'MarkerEdgeColor','none', 'MarkerFaceColor','y');
set(h_pca(1).MarkerHandle, 'FaceColorType','truecoloralpha', 'FaceColorData',uint8(255*[0;0;1;0.3]));
set(h_pca(2).MarkerHandle, 'FaceColorType','truecoloralpha', 'FaceColorData',uint8(255*[1;0;0;0.3]));
set(h_pca(3).MarkerHandle, 'FaceColorType','truecoloralpha', 'FaceColorData',uint8(255*[1;1;0;0.3]));
title('2D PCA reduction of raw data')
xlabel('PCA1') 
ylabel('PCA2')
% t-SNE on raw data
feature_tsne = tsne(data_compress);
h_tsne = gscatter(feature_tsne(:,1),feature_tsne(:,2),labels_0,[],[],10);
title('t-SNE reduction of raw data')
xlabel('t-SNE 1') 
ylabel('t-SNE 2')

%image generation
mkdir(fullfile('data_images/MSI/bladder'));
mkdir(fullfile('data_images/MSI/bladder','0'));
mkdir(fullfile('data_images/MSI/bladder','1'));
mkdir(fullfile('data_images/MSI/bladder','2'));
mkdir(fullfile('data_images/MSI/bladder','3'));
for ii=1:size(data, 1)
    newx = data(ii, :);

    %shift left 5m/z
    %noisyregion = newx(59591:60000);
    %x = randi(size(noisyregion,1),1,500);
    %newintensities = noisyregion(x);
    %newx = [newx(501:60000),newintensities];

    %shift right 0.5m/z
    %noisyregion = newx(1:50);
    %x = randi(size(noisyregion,1),1,50);
    %newintensities = noisyregion(x);
    %newx = [newintensities,newx(1:59950)];

    %newx = reshape(newx, 10, 6000);
    %newx = sum(newx, 1) * 0.1;
    %newx = reshape(newx,6000,1);

    [cfs,f] = cwt(newx,'bump',2000,'VoicePerOctave', 48);
    im = abs(cfs);
    im = im / max(im(:)) * 255;
    im = uint8(im);
    im = histeq(im);
    im = ind2rgb((im),jet(256));
    imgLoc = fullfile('data_images/MSI/bladder', num2str(labels(ii)));
    imFileName = strcat(num2str(ii),'.jpg');
    imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
end

% CNN training
allImages = imageDatastore(fullfile('data_images/MSI/bladder'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
[imgsTrain,imgsValidation,imgsTest] = splitEachLabel(allImages,0.7,'randomized');
disp(['Number of training images: ',num2str(numel(imgsTrain.Files))]);
disp(['Number of validation images: ',num2str(numel(imgsValidation.Files))]);
disp(['Number of test images: ',num2str(numel(imgsTest.Files))]);
numClasses = numel(categories(imgsTrain.Labels));

% google net
gnet = googlenet;
glgraph = layerGraph(gnet);
newDropoutLayer1 = dropoutLayer(0.4,'Name','new_Dropout');
glgraph = replaceLayer(glgraph,'pool5-drop_7x7_s1',newDropoutLayer1);
gnewConnectedLayer = fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
glgraph = replaceLayer(glgraph,'loss3-classifier',gnewConnectedLayer);
gnewClassLayer = classificationLayer('Name','new_classoutput');
glgraph = replaceLayer(glgraph,'output',gnewClassLayer);
%plot(glgraph);
%analyzeNetwork(glgraph);
options = trainingOptions('adam',...
'MaxEpochs',3,...
'InitialLearnRate',1e-5,...
'ValidationData',imgsValidation,...
'ValidationFrequency',50);

% train
gtrainedGN = trainNetwork(imgsTrain,glgraph,options);
[YPredgtrain,probsgtrain] = classify(gtrainedGN,imgsTrain);
[YPredgval,probsgval] = classify(gtrainedGN,imgsValidation);
[YPredgtest,probsgtest] = classify(gtrainedGN,imgsTest);
[Xgtrain1,Ygtrain1,Tgtrain1,AUCgtrain1] = perfcurve(imgsTrain.Labels,probsgtrain(:, 1),1);
[Xgval1,Ygval1,Tgval1,AUCgval1] = perfcurve(imgsValidation.Labels,probsgval(:, 1),1);
[Xgtest1,Ygtest1,Tgtest1,AUCgtest1] = perfcurve(imgsTest.Labels,probsgtest(:, 1),1);
disp(['AUC of label 1 vs others is : ',num2str(AUCg1)])

% resnet
rnet = resnet50;
rlgraph = layerGraph(rnet);
rnewDropoutLayer = dropoutLayer(0.4,'Name','new_Dropout');
rlgraph = addLayers(rlgraph,rnewDropoutLayer);
rlgraph = disconnectLayers(rlgraph,'avg_pool','fc1000');
rlgraph = connectLayers(rlgraph,'avg_pool','new_Dropout');
rlgraph = connectLayers(rlgraph,'new_Dropout','fc1000');
rnewConnectedLayer = fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
rlgraph = replaceLayer(rlgraph,'fc1000',rnewConnectedLayer);
rnewClassLayer = classificationLayer('Name','new_classoutput');
rlgraph = replaceLayer(rlgraph,'ClassificationLayer_fc1000',rnewClassLayer);
%analyzeNetwork(rlgraph);
options = trainingOptions('adam',...
'MaxEpochs',1,...
'InitialLearnRate',2e-5,...
'ValidationData',imgsValidation,...
'ValidationFrequency',50);

% train
rtrainedGN = trainNetwork(imgsTrain,rlgraph,options);
[YPredrtrain,probsrtrain] = classify(rtrainedGN,imgsTrain);
[YPredrval,probsrval] = classify(rtrainedGN,imgsValidation);
[YPredrtest,probsrtest] = classify(rtrainedGN,imgsTest);
[Xr2,Yr2,Tr2,AUCr2] = perfcurve(imgsTest.Labels,probsrtest(:, 2),2);
disp(['AUC of label 2 vs others is : ',num2str(AUCr2)])

annotation = zeros(134,260);
for i=1:numel(imgsTrain.Files)
    [filepath,name,ext] = fileparts(imgsTrain.Files(i));
    coords = xy_coords(str2double(name),:);
    annotation(coords(2),coords(1)) = double(YPredgtrain(i));
end
for i=1:numel(imgsTest.Files)
    [filepath,name,ext] = fileparts(imgsTest.Files(i));
    coords = xy_coords(str2double(name),:);
    annotation(coords(2),coords(1)) = double(YPredgtest(i));
end
h=heatmap(annotation);
set(gcf,'position',[100,100,260*2,134*2])
h.ColorbarVisible = 'off';
h.Colormap = parula;
XLabels = h.XDisplayLabels;
h.XDisplayLabels = repmat(' ',size(XLabels,1), size(XLabels,2));
YLabels = h.YDisplayLabels;
h.YDisplayLabels = repmat(' ',size(YLabels,1), size(YLabels,2));
h.Title = 'ResNet-50 predicted labels';
%XLabels = 1:260;
%CustomXLabels = string(XLabels);
%CustomXLabels(mod(XLabels,50) ~= 0) = " ";
%h.XDisplayLabels = CustomXLabels;
%YLabels = 1:134;
%CustomYLabels = string(YLabels);
%CustomYLabels(mod(YLabels,20) ~= 0) = " ";
%h.YDisplayLabels = CustomYLabels;
%set(gcf,'position',[100,100,260*2,134*2])
cm = confusionmat(imgsTest.Labels,YPred1);
plotconfusion(imgsTest.Labels,YPred1);
histogram(probs1((imgsTest.Labels=='2'),1));
hold on
histogram(probs1((imgsTest.Labels~='2'),1));
hold off

analyzeNetwork(trainedGN1)
featuretest = activations(trainedGN1,imgsValidation,'pool5-7x7_s1');
sz = size(featuretest);
featuretest = reshape(featuretest,[sz(3) sz(4)]);
featuretest = transpose(featuretest);
feature_tsne = tsne(featuretest);
gscatter(feature_tsne(:,1),feature_tsne(:,2),imgsValidation.Labels,[],[],20)
title('t-SNE plot of extracted imaging features')
xlabel('t-SNE 1') 
ylabel('t-SNE 2') 

% linear SVM/Random Forest
rng(1)
cv = cvpartition(size(data,1),'HoldOut',0.4);
idx = cv.training;
data_train = data(idx,:);
data_test  = data(~idx,:);
labels_train = labels(idx,:);
labels_test = labels(~idx,:);
Mdlsvm = fitcecoc(data_train,labels_train,'Coding','onevsall','FitPosterior',true,'Prior','uniform');
tTree = templateTree('surrogate','on');
tEnsemble = templateEnsemble('GentleBoost',100,tTree);
options = statset('UseParallel',true);
Mdlt = fitcecoc(data_train,labels_train,'Coding','onevsall','FitPosterior',true, ...
                'Learners',tEnsemble,'NumBins',10,'Prior','uniform','Options',options);
[labels_svm_train,~,~,Posterior_svm_train] = predict(Mdlsvm,data_train);
[labels_svm_test,~,~,Posterior_svm_test] = predict(Mdlsvm,data_test);
[labels_t_train,~,~,Posterior_t_train] = predict(Mdlt,data_train);
[labels_t_test,~,~,Posterior_t_test] = predict(Mdlt,data_test);
[Xt3_,Yt3_,Tt3_,AUCt3_] = perfcurve(labels_test,Posterior_t_test(:, 3),3);

annotation = zeros(134,260);
coords = xy_coords(~idx,:);
for i=1:size(data_test,1)
    annotation(coords(i,2),coords(i,1)) = double(labels_svm_test(i));
end
coords = xy_coords(idx,:);
for i=1:size(data_train,1)
    annotation(coords(i,2),coords(i,1)) = double(labels_svm_train(i));
end
h=heatmap(annotation);

annotation = transpose(annotation);
annotation = reshape(annotation,134*260,1);
annotation = annotation(~(annotation==0),:);
annotation = [annotation;ones(16,1)];
for ii=1:size(labels_0)
    annotation = [annotation(1:labels_0(ii)-1);0;annotation(labels_0(ii):end)];
end
annotation = annotation(1:134*260);
annotation = reshape(annotation,260,134);
annotation = transpose(annotation);

precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';
accuracy = @(confusionMat) sum(diag(confusionMat))/sum(confusionMat,'all');


%% LCMS dataset
label_lcms_ = readcell('data/EPL/Sample List Details.xlsx','Sheet',2);
label_lcms_ = label_lcms_(1:end-1,1:2);
label_lcms = {};
for ii = 1:length(label_lcms_)
    if startsWith(label_lcms_{ii,2},"P")
        label_lcms{end+1,1} = label_lcms_{ii,1}(end-2:end);
        label_lcms{end,2} = "P";
    end
    if startsWith(label_lcms_{ii,2},"NP")
        label_lcms{end+1,1} = label_lcms_{ii,1}(end-2:end);
        label_lcms{end,2} = "NP";
    end
    if startsWith(label_lcms_{ii,2},"EL")
        label_lcms{end+1,1} = label_lcms_{ii,1}(end-2:end);
        label_lcms{end,2} = "EL";
    end
    if startsWith(label_lcms_{ii,2},"LM")
        label_lcms{end+1,1} = label_lcms_{ii,1}(end-2:end);
        label_lcms{end,2} = "LM";
    end
end

%image generation
filePattern = fullfile('data/EPL/mz_778/*.csv');
theFiles = dir(filePattern);
imgLoc = fullfile('data_images/EPL/mz_778');
for k = 1:length(theFiles)
    data = importdata(fullfile('data/EPL/mz_778',theFiles(k).name));
    data = data(:,2);
    [cfs,f] = cwt(sqrt(data),'bump',2000,'VoicePerOctave', 48);
    im = abs(cfs);
    im = im / max(im(:)) * 65535;
    im = uint16(im);
    im = histeq(im);
    im = ind2rgb((im),jet(65536));
    name = theFiles(k).name(end-6:end-4);
    label = label_lcms{[label_lcms{:}]==name,2};
    imgLoc = fullfile('data_images/EPL/mz_778/bump',label);
    imFileName = strcat(name,'.jpg');
    imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
end

%CNN training
allImages = imageDatastore(fullfile('bump_pnp'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
rng default
[imgsTrain2,imgsTest2] = splitEachLabel(allImages,0.7,'randomized');
numClasses = numel(categories(imgsTrain2.Labels));
options = trainingOptions('adam',...
'MiniBatchSize',64,...
'MaxEpochs',15,...
'InitialLearnRate',2e-5,...
'ValidationData',imgsTest2,...
'ValidationFrequency',10);
% googlenet
net2 = googlenet;
lgraph2 = layerGraph(net2);
newDropoutLayer2 = dropoutLayer(0.6,'Name','new_Dropout');
lgraph2 = replaceLayer(lgraph2,'pool5-drop_7x7_s1',newDropoutLayer2);
newConnectedLayer2 = fullyConnectedLayer(numClasses,'Name','new_fc',...
    'WeightLearnRateFactor',5,'BiasLearnRateFactor',5);
lgraph2 = replaceLayer(lgraph2,'loss3-classifier',newConnectedLayer2);
newClassLayer2 = classificationLayer('Name','new_classoutput');
lgraph2 = replaceLayer(lgraph2,'output',newClassLayer2);
rng default
trainedGN2 = trainNetwork(imgsTrain2,lgraph2,options);
[YPred2,probs2] = classify(trainedGN2,imgsTest2);
[~,~,~,AUC2] = perfcurve(imgsTest2.Labels,probs2(:, 1),'NP');
disp(['AUC of model googlenetmexh2 is : ',num2str(AUC2)])
cm_2 = confusionmat(imgsTest2.Labels,YPred2);
precision_2 = cm_2(2,2)/(cm_2(2,2)+cm_2(1,2));
recall_2 = cm_2(2,2)/(cm_2(2,2)+cm_2(2,1));
accuracy_2 = (cm_2(1,1)+cm_2(2,2))/sum(cm_2,'all');

