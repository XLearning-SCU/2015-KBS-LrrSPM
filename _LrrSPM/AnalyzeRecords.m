% =========================================================================
% An example code for the algorithm proposed in

% Xi Peng, Rui Yan, Bo Zhao, Huajin Tang, and Zhang Yi,
% Fast Low-rank Representation based Spatial Pyramid Matching for Image Classification,
% Knowledge based Systems, Volume 90, December 2015, Pages 14-22.

%    
% If the codes or data sets are helpful to you, please appropriately cite our works. Thank you very much!
% Written by Xi Peng @ I2R A*STAR,
% more information can be found from my website: http://machineilab.org/users/pengxi/ or www.pengxi.me
% Augest, 2014.
% =========================================================================
clc;
fprintf('\n=====================Linear SVM==========================\n');
% fprintf('Max classification accuracy: %f\n', max(linear_accuracy)*100);
fprintf('Average classification accuracy: %f\n', mean(linear_accuracy)*100);
fprintf('STD classification accuracy: %f\n', std(linear_accuracy)*100);
fprintf('Time for coding: %f\n', mean(time_coding));
fprintf('Time for classification: %f\n', mean(time_linearSVM));
