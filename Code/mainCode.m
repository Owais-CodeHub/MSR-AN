% 2D/3D Radiographic Content-based Retrieval and Analysis (MSR-AN by Owais et al.,)

clc;
clear all;
close all;

gpuDevice(1);
reset(gpuDevice());

load('model_1_MSA_SN.mat')
model_1_MSA_SN = F1.TrainedModel;

load('model_2_RFA_SN.mat');
model_2_RFA_SN = F1.TrainedModel;

imdsQuery = imageDatastore('Query DB', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsRetieval = imageDatastore('Retrieval DB', 'IncludeSubfolders', true, 'LabelSource', 'foldernames'); 

queryImg = 1:4;   % e.g., use 5 for 2D, use 5:7 for 3D, its diseased 

topK = 5;
stepSizeNew = 16; 

allResults = [];

%% Retrieval database feature extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

WholeDatasetFeatureVector = [];
WholeDatasetFeatureVector = fn_extractDeepFeaturesForLSTM(model_1_MSA_SN,'dropout',imdsRetieval,224);
[imdsTrainData,imdsTrainLabels] = fn_adjustLSTM_Data_Equal_Samples(WholeDatasetFeatureVector, stepSizeNew);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Query feature extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
WholeDatasetFeatureVector = [];
WholeDatasetFeatureVector = fn_extractDeepFeaturesForLSTM(model_1_MSA_SN,'dropout',imdsQuery,224);
[imdsTestData,imdsTestLabels] = fn_adjustLSTM_Data_Equal_Samples(WholeDatasetFeatureVector, stepSizeNew);
WholeDatasetFeatureVector = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Retrieval activation features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
actTrain = [];

for i = 1:size(imdsTrainData,1)
    i
    temp = activations(model_2_RFA_SN, imdsTrainData{i}, 'dropout7','OutputAs','channels');
    actTrain = [actTrain; temp'];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Query activation feature: 2D or 3D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
actQuery = [];

for q = 1:length(queryImg)

    temp = activations(model_2_RFA_SN, imdsTestData{queryImg(q)}, ...
        'dropout7','OutputAs','channels');

    actQuery = [actQuery; temp'];
end

tempQuery = mean(actQuery,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Matching
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[D,loc] = maxk(1 - pdist2(actTrain,tempQuery,'cosine'), topK);

[pred,probs] = classify(model_2_RFA_SN, imdsTestData{queryImg(1)});
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Label conversion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
actualLabel = getClassName(double(imdsTestLabels(queryImg(1))));
predLabel   = getClassName(double(pred));

imgData = imdsRetieval.Files;
imgLabel = imdsRetieval.Labels;

retrivedImg = imgData(loc);
retrievedLab = imgLabel(loc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GUI window
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = figure('Name','Medical Image Retrieval GUI', ...
    'Color',[0.96 0.97 0.99], ...
    'Position',[80 80 1550 780]);

annotation(fig,'textbox',[0.02 0.93 0.96 0.055], ...
    'String','2D/3D Radiographic Content-based Retrieval and Analysis (MSR-AN by Owais et al.,)', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontSize',22, ...
    'FontWeight','bold', ...
    'Color',[0.05 0.15 0.35]);

annotation(fig,'textbox',[0.02 0.885 0.96 0.035], ...
    'String',['Actual Class: ', actualLabel, ...
              '    |    Predicted Class: ', predLabel, ...
              '    |    Top-K Retrieved Results: ', num2str(topK)], ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontSize',13, ...
    'FontWeight','bold', ...
    'Color',[0.15 0.15 0.15]);

%% Query panel

queryPanel = uipanel('Parent',fig, ...
    'Title','QUERY INPUT', ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'ForegroundColor',[0 0.2 0.7], ...
    'BackgroundColor','white', ...
    'Position',[0.03 0.15 0.27 0.70]);

axQ = axes('Parent',queryPanel, ...
    'Position',[0.08 0.22 0.84 0.70]);

if length(queryImg) == 1

    imshow(imread(imdsQuery.Files{queryImg}), [], 'Parent', axQ);
    queryType = '2D Query Scan';

else

    montage(imdsQuery.Files(queryImg), ...
        'Size',[1 length(queryImg)], ...
        'Parent',axQ);

    queryType = ['3D Query Scan: Slices ', ...
        num2str(queryImg(1)), ' to ', num2str(queryImg(end))];
end

title(axQ,queryType, ...
    'FontSize',12, ...
    'FontWeight','bold');

uicontrol('Parent',queryPanel, ...
    'Style','text', ...
    'String',{['Actual: ', actualLabel], ['Predicted: ', predLabel]}, ...
    'Units','normalized', ...
    'Position',[0.05 0.03 0.90 0.13], ...
    'BackgroundColor','white', ...
    'ForegroundColor',[0 0.2 0.7], ...
    'FontSize',12, ...
    'FontWeight','bold');

%% Retrieval results panel

resultPanel = uipanel('Parent',fig, ...
    'Title','TOP RETRIEVED MATCHES', ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'ForegroundColor',[0.1 0.35 0.1], ...
    'BackgroundColor','white', ...
    'Position',[0.32 0.38 0.65 0.47]);

for i = 1:topK

    leftPos = 0.02 + (i-1)*0.195;

    axR = axes('Parent',resultPanel, ...
        'Position',[leftPos 0.27 0.17 0.62]);

    imshow(imread(retrivedImg{i}), [], 'Parent', axR);

    retrievedLabelName = getClassName(double(retrievedLab(i)));

    title(axR,{['Rank #', num2str(i)], retrievedLabelName}, ...
        'FontSize',10, ...
        'FontWeight','bold');

    uicontrol('Parent',resultPanel, ...
        'Style','text', ...
        'String',['Similarity: ', num2str(D(i),'%.3f')], ...
        'Units','normalized', ...
        'Position',[leftPos 0.07 0.17 0.10], ...
        'BackgroundColor','white', ...
        'ForegroundColor',[0.15 0.15 0.15], ...
        'FontSize',10, ...
        'FontWeight','bold');
end

%% Similarity score panel

scorePanel = uipanel('Parent',fig, ...
    'Title','SIMILARITY SCORE ANALYSIS', ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'ForegroundColor',[0.45 0.20 0.05], ...
    'BackgroundColor','white', ...
    'Position',[0.32 0.15 0.65 0.20]);

axB = axes('Parent',scorePanel, ...
    'Position',[0.08 0.22 0.88 0.65]);

bar(axB,D,0.55);
ylim(axB,[0 1]);
grid(axB,'on');

xlabel(axB,'Retrieved Rank','FontWeight','bold');
ylabel(axB,'Cosine Similarity','FontWeight','bold');
title(axB,'Top-K Matching Scores','FontWeight','bold');

xticks(axB,1:topK);
xticklabels(axB,"Rank " + string(1:topK));

for i = 1:topK
    text(axB,i,D(i)+0.03,num2str(D(i),'%.3f'), ...
        'HorizontalAlignment','center', ...
        'FontSize',10, ...
        'FontWeight','bold');
end

%% Bottom summary

annotation(fig,'textbox',[0.03 0.04 0.94 0.06], ...
    'String',['Summary: ', queryType, ...
              '  |  Actual = ', actualLabel, ...
              '  |  Predicted = ', predLabel, ...
              '  |  Best Retrieved Match = ', getClassName(double(retrievedLab(1))), ...
              '  |  Best Score = ', num2str(D(1),'%.4f')], ...
    'BackgroundColor',[1 1 1], ...
    'EdgeColor',[0.75 0.75 0.75], ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'Color',[0.05 0.05 0.05]);

%% Print results in command window

disp('============== Retrieval Results ==============')
disp(['Actual Query Class: ', actualLabel])
disp(['Predicted Query Class: ', predLabel])

for i = 1:topK
    fprintf('Rank %d | Retrieved Class: %s | Score: %.4f | File: %s\n', ...
        i, getClassName(double(retrievedLab(i))), D(i), retrivedImg{i});
end

%% Local function for class names

function className = getClassName(labelValue)

    if labelValue == 1
        className = 'Infected';
    elseif labelValue == 2
        className = 'Normal';
    else
        className = ['Unknown Class ', num2str(labelValue)];
    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                             % Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function combinedImage = CAMshow(im,CAM)

imSize = size(im);
CAM = imresize(CAM,imSize(1:2));
CAM = normalizeImage(CAM);
CAM(CAM<0.1) = 0;
cmap = jet(255).*linspace(0,1,255)';
CAM = ind2rgb(uint8(CAM*255),cmap)*255;
combinedImage = double((im)./2) + CAM;
combinedImage = normalizeImage(combinedImage)*255;
combinedImage = uint8(combinedImage);


end

function N = normalizeImage(I)
minimum = min(I(:));
maximum = max(I(:));
N = (I-minimum)/(maximum-minimum);
end



function allResults = getResults(pred,Actual_Label)



Pred_Actual_Label = [pred Actual_Label];
[c_matrix,Result] = confusionmat(Actual_Label,pred);
[c_matrix_per,Avg_Accuracy1,Avg_F1_Score1,Avg_Precision1,Sensivity1_Recal1] = fn_a_PCA_Acc_F1_Pr_Re(c_matrix);

Acc_F1_mAP_mAR_AUC = [Avg_Accuracy1,Avg_F1_Score1,Avg_Precision1,Sensivity1_Recal1];


TPR = c_matrix(1,1)./(c_matrix(1,1)+c_matrix(1,2)); 
TNR = c_matrix(2,2)./(c_matrix(2,1)+c_matrix(2,2));



allResults = [TPR TNR Acc_F1_mAP_mAR_AUC];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% 1) Extract Deep Features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function WholeDatasetFeatureVector = fn_extractDeepFeaturesForLSTM(net,layer_name,imdsTrain,newImgSize)
class_names=unique(imdsTrain.Labels)
WholeDatasetFeatureVector = []; % matrix array for storing normalized feature vectors for whole class and whole dataset
for idx=1:length(class_names) % loop forselecting class 
    image_idx = find(imdsTrain.Labels==class_names(idx));
    act1NoNormalize = [];
    for idx2 = 1:length(image_idx) % loop for selecting class image
        simulation_progress_L_C_I = [idx idx2]
        im = imread(imdsTrain.Files{image_idx(idx2)});
        im2 = [];
        if size(im,3)~=3
            im2(:,:,1) = imresize(im,[newImgSize newImgSize],'nearest');
            im2(:,:,2) = imresize(im,[newImgSize newImgSize],'nearest');
            im2(:,:,3) = imresize(im,[newImgSize newImgSize],'nearest');
            im=[]; im=im2;
        else
            im=imresize(im,[newImgSize,newImgSize],'nearest');
        end
        act1 = activations(net, im, layer_name,'OutputAs','channels');
        
        act1= reshape(act1,[size(act1,1)*size(act1,2)*size(act1,3),1]);

        
        act1NoNormalize =[act1NoNormalize; act1'];
    end
    WholeDatasetFeatureVector{idx} = act1NoNormalize;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 2) Merge 4 Datastore
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mergeData = fn_MergeDatastore_4_DB(imd1,imd2,imd3,imd4)
mergeData = splitEachLabel(imd1,1.0,'randomize');
Files = [imd1.Files;imd2.Files;imd3.Files;imd4.Files];
Labels = [imd1.Labels;imd2.Labels;imd3.Labels;imd4.Labels];
mergeData.Files=Files;
mergeData.Labels=Labels;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 3) Merge 2 Datastore
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mergeData = fn_MergeDatastore_2_DB(imd1,imd2)

mergeData = splitEachLabel(imd1,1.0,'randomize');
Files = [imd1.Files;imd2.Files];
Labels = [imd1.Labels;imd2.Labels];
mergeData.Files=Files;
mergeData.Labels=Labels;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 4) Adjust CNN features for LSTM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Data,Labels] = fn_adjustLSTM_Data_Equal_Samples(Train , stepSizeN)
   VidOpt={'Y';'Y'};


  % Label Sequence: 1<<Covid19 Pneumonia Axial View>>  2<<Covid19 Pneumonia X-Ray>>  3<<Normal Axial View>>  4<<Normal X-Ray>> 



% stepSize:  number of total successive frames t, t-1, t-2 ... etc
Data = [];
Labels = [];
 idx = 1;
for i = 1 : length(Train)
    if(strcmp(VidOpt{i},'N')) || (stepSizeN==1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    stepSize = 1;

    temp1 = Train{i};
    temp1 = temp1';
    for j = stepSize : size(temp1,2)
    progress = [i,j];
    Data{idx} = temp1(:,j-stepSize+1 : j);
    Labels = [Labels ; i];
    idx = idx + 1;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    stepSize = stepSizeN;
    
    temp1 = Train{i};
    temp1 = temp1';
    if (size(temp1,2)<stepSizeN) stepSize=size(temp1,2); end
    
    temp2 = temp1(:,1:stepSize-1);
    for kk = 1 : stepSize-1
        Data{idx} = temp2(:,1:kk);             
        Labels = [Labels ; i];
        idx = idx + 1;
    end
    
    for j = stepSize : size(temp1,2)
        progress = [i,j];
        Data{idx} = temp1(:,j-stepSize+1 : j);
        Labels = [Labels ; i];
        idx = idx + 1;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end
Data = Data';
Labels = categorical(Labels);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%