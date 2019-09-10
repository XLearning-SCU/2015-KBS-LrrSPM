% ========================================================================
% Pooling the fast LRR codes to form the image feature
% USAGE: [beta] = LrrSPM_pooling(feaSet, ProjM, pyramid, knn)
% Inputs
%       feaSet      -the coordinated local descriptors
%       ProjM       -the projection matrix for fast LRR coding
%       pyramid     -the spatial pyramid structure
%       knn         -the number of neighbors for llc coding
%       lambda      -the regularization parameter for fast LRR
% Outputs
%       beta        -the output image feature
%
% Written by Xi Peng @ I2R A*STAR
% Apr., 2014
% ========================================================================

function [beta] = LrrSPM_pooling(feaSet, ProjM, pyramid, EngRatio)

dSize = size(ProjM, 1);
nSmp = size(feaSet.feaArr, 2);

img_width = feaSet.width;
img_height = feaSet.height;
idxBin = zeros(nSmp, 1);

% fast LRR coding
fLRR_codes = ProjM*feaSet.feaArr;
if EngRatio>0
    engery = fLRR_codes./ repmat(sum(fLRR_codes),size(fLRR_codes,1),1);
    fLRR_codes(engery<(EngRatio/size(fLRR_codes,1)))=0;
end

% spatial levels
pLevels = length(pyramid);
% spatial bins on each level
pBins = pyramid.^2;
% total spatial bins
tBins = sum(pBins);

beta = zeros(dSize, tBins);
bId = 0;

for iter1 = 1:pLevels,
    
    nBins = pBins(iter1);
    
    wUnit = img_width / pyramid(iter1);
    hUnit = img_height / pyramid(iter1);
    
    % find to which spatial bin each local descriptor belongs
    xBin = ceil(feaSet.x / wUnit);
    yBin = ceil(feaSet.y / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    
    for iter2 = 1:nBins,     
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end
        beta(:, bId) = max(abs(fLRR_codes(:, sidxBin)), [], 2);
    end
end

if bId ~= tBins,
    error('Index number error!');
end

beta = beta(:);
beta = beta./sqrt(sum(beta.^2));
