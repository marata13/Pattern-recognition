clear
clc
%DATA PREPROCESSING
%importing the data table
Data = readtable('housing.csv');
Data.Properties;

Ocean_proximity = Data.ocean_proximity;

%DATA VISUALIZATION PART 1
%plotting the ocean_proximity values in one graph
figure;
histogram(categorical(Ocean_proximity));
title(Data.Properties.VariableNames{1,10});
ylabel('Frequency');

%plotting the arithmetical variables
NewData = table();
for y=1:9
    NewData(:,y) = Data(:,y);
end
NewData = table2array(NewData);
figure;
for n=1:9
    subplot(3,3,n);
    H1 = histogram(NewData(:,n));
    title(Data.Properties.VariableNames{1,n});
    ylabel('Frequency');
end
%END OF DATA VISUALIZATION PART 1

%scaling the arithmetic data
Data.longitude = (Data.longitude - min(Data.longitude)) / (max(Data.longitude) - min(Data.longitude));
Data.latitude = (Data.latitude - min(Data.latitude)) / (max(Data.latitude) - min(Data.latitude));
Data.housing_median_age = (Data.housing_median_age - min(Data.housing_median_age)) / (max(Data.housing_median_age) - min(Data.housing_median_age));
Data.total_rooms = (Data.total_rooms - min(Data.total_rooms)) / (max(Data.total_rooms) - min(Data.total_rooms));
Data.total_bedrooms = (Data.total_bedrooms - min(Data.total_bedrooms)) / (max(Data.total_bedrooms) - min(Data.total_bedrooms));
Data.population = (Data.population - min(Data.population)) / (max(Data.population) - min(Data.population));
Data.households = (Data.households - min(Data.households)) / (max(Data.households) - min(Data.households));
Data.median_income = (Data.median_income - min(Data.median_income)) / (max(Data.median_income) - min(Data.median_income));

%encoding the categorical data
Data.ocean_proximity = onehotencode(categorical(Data.ocean_proximity),2);

%converting data table into matrix
Formatted_Data = table2array(Data);

%replacing NaN values
for k=1:8
    column = Formatted_Data(:,k);
    column = column';
    M = median(column,'omitnan');
    for z=1:length(column)
        if (isnan(Formatted_Data(z,k)))
            Formatted_Data(z,k) = M;
        end
    end
end

%DATA VISUALIZATION PART 2
%data combinations plots
%2 VARIABLES PLOTS
%No1: housing_median age, median_house_value
figure;
plot(Formatted_Data(:,3), Formatted_Data(:,9), 'o', 'Linewidth',2);
xlabel('Housing_median_age');
ylabel('Median_house_value');

%No2: median_income and population
figure;
plot(Formatted_Data(:,8), Formatted_Data(:,6), 'o', 'Linewidth',2);
xlabel('Median_income');
ylabel('Population');

%No3: population and households
figure;
plot(Formatted_Data(:,6), Formatted_Data(:,7), 'o', 'Linewidth',2);
xlabel('Population');
ylabel('Households');

%3 VARIABLES PLOTS
%No1: total_rooms, total_bedroooms, population
figure;
scatter(Formatted_Data(:,4), Formatted_Data(:,5), 10, Formatted_Data(:,6), 'filled');
c1 = colorbar;
colormap(turbo);
xlabel('total_rooms');
ylabel('total_bedrooms');
c1.Label.String = 'population';

%No2: Median_income, median_house_value, ocean_proximity
figure;
scatter(Formatted_Data(:,8), Formatted_Data(:,9), 10, categorical(Ocean_proximity), 'filled');
c2 = colorbar;
colormap(turbo);
xlabel('median_income');
ylabel('median_house_value');
c2.Label.String = 'ocean_proximity';


%4 VARIABLES PLOTS
%No1: population, median_income, total_rooms, housing_median_age
%Note: When the housing median age increases, the size of the points increases, as
%well. The colorbar represents the total rooms.
figure;
sz = 10 + ((Formatted_Data(:,3) - min(Formatted_Data(:,3))) * (100-10)) / (max(Formatted_Data(:,3)) - min(Formatted_Data(:,3)));
scatter(Formatted_Data(:,6), Formatted_Data(:,8), sz, Formatted_Data(:,4));
c3 = colorbar;
colormap(turbo);
xlabel('population');
ylabel('median_income');
c3.Label.String = 'total_rooms';

%No2: longitude, latitude, ocean_proximity, median_house_value
%Note: When the house value increases, the size of the points increases, as
%well. The colorbar represents the ocean proximity of each point.
figure;
sz = 10 + ((Formatted_Data(:,9) - min(Formatted_Data(:,9))) * (200-10)) / (max(Formatted_Data(:,9)) - min(Formatted_Data(:,9)));
scatter(Formatted_Data(:,1), Formatted_Data(:,2), sz, categorical(Ocean_proximity));
c4 = colorbar;
colormap(turbo);
xlabel('longitude');
ylabel('latitude');
c4.Label.String = 'ocean_proximity';

%REGRESSION ANALYSIS


y = Formatted_Data(:,9);
x = [Formatted_Data(:,1) Formatted_Data(:,2) Formatted_Data(:,3) Formatted_Data(:,4) Formatted_Data(:,5) Formatted_Data(:,6) Formatted_Data(:,7) Formatted_Data(:,8) Formatted_Data(:,10) Formatted_Data(:,11) Formatted_Data(:,12) Formatted_Data(:,13) Formatted_Data(:,14)];
x = [ones(length(x),1) x];

%LEAST SQUARES
LeastSquares_meanSquareErrorTrained = 0;
LeastSquares_meanAbsoluteErrorTrained = 0;
LeastSquares_meanSquareErrorTest = 0;
LeastSquares_meanAbsolutErrorTest = 0;
for k = 1:10
    
    test_data = x((k-1)*length(x)/10+1:(k)*length(x)/10,:);
    ytest = y((k-1)*length(y)/10+1:(k)*length(y)/10,:);
    if k == 1
        train_data = x((k)*length(x)/10+1:length(x),:);
        ytrain = y((k)*length(y)/10+1:length(y),:);
    elseif k==10
       train_data = x(1:9*length(x)/10,:);
       ytrain = y(1:9*length(y)/10,:);
    else
        train_data = [x(1:(k-1)*length(x)/10,:); x(k*length(x)/10+1:length(x),:)];
        ytrain = [y(1:(k-1)*length(y)/10,:); y(k*length(y)/10+1:length(y),:)];
    end
    
    weight = pinv(train_data)*ytrain;
    
    LeastSquares_meanSquareErrorTrained = LeastSquares_meanSquareErrorTrained + sum((ytrain-train_data*weight).^2)/length(ytrain);
    LeastSquares_meanAbsoluteErrorTrained = LeastSquares_meanAbsoluteErrorTrained + sum(abs(ytrain -train_data*weight))/length(ytrain);
    LeastSquares_meanSquareErrorTest = LeastSquares_meanSquareErrorTest + sum((ytest-test_data*weight).^2)/length(ytest);
    LeastSquares_meanAbsolutErrorTest = LeastSquares_meanAbsolutErrorTest + sum(abs(ytest -test_data*weight))/length(ytest);
    
end
LeastSquares_meanSquareErrorTrained=LeastSquares_meanSquareErrorTrained/10
LeastSquares_meanAbsoluteErrorTrained=LeastSquares_meanAbsoluteErrorTrained/10
LeastSquares_meanSquareErrorTest=LeastSquares_meanSquareErrorTest/10
LeastSquares_meanAbsolutErrorTest=LeastSquares_meanAbsolutErrorTest/10



%LEAST MEAN SQUARES
LeastMeanSquares_meanSquareErrorTrained = 0;
LeastMeanSquares_meanAbsoluteErrorTrained = 0;
LeastMeanSquares_meanSquareErrorTest = 0;
LeastMeanSquares_meanAbsolutErrorTest = 0;
for k = 1:10
    
    test_data = x((k-1)*length(x)/10+1:(k)*length(x)/10,:);
    ytest = y((k-1)*length(y)/10+1:(k)*length(y)/10,:);
    if k == 1
        train_data = x((k)*length(x)/10+1:length(x),:);
        ytrain = y((k)*length(y)/10+1:length(y),:);
    elseif k==10
       train_data = x(1:9*length(x)/10,:);
       ytrain = y(1:9*length(y)/10,:);
    else
        train_data = [x(1:(k-1)*length(x)/10,:); x(k*length(x)/10+1:length(x),:)];
        ytrain = [y(1:(k-1)*length(y)/10,:); y(k*length(y)/10+1:length(y),:)];
    end
    
    weight = zeros(14,1);
    for i = 1:length(ytrain)
       weight = weight + (1/i)*train_data(i,:)'*(ytrain(i)-train_data(i,:)*weight);
    end
    
    LeastMeanSquares_meanSquareErrorTrained = LeastSquares_meanSquareErrorTrained + sum((ytrain-train_data*weight).^2)/length(ytrain);
    LeastMeanSquares_meanAbsoluteErrorTrained = LeastSquares_meanAbsoluteErrorTrained + sum(abs(ytrain -train_data*weight))/length(ytrain);
    LeastMeanSquares_meanSquareErrorTest = LeastSquares_meanSquareErrorTest + sum((ytest-test_data*weight).^2)/length(ytest);
    LeastMeanSquares_meanAbsolutErrorTest = LeastSquares_meanAbsolutErrorTest + sum(abs(ytest -test_data*weight))/length(ytest);
    
end

LeastMeanSquares_meanSquareErrorTrained=LeastMeanSquares_meanSquareErrorTrained/10
LeastMeanSquares_meanAbsoluteErrorTrained=LeastMeanSquares_meanAbsoluteErrorTrained/10
LeastMeanSquares_meanSquareErrorTest=LeastMeanSquares_meanSquareErrorTest/10
LeastMeanSquares_meanAbsolutErrorTest=LeastMeanSquares_meanAbsolutErrorTest/10



%Multilayer Neural Network
MultilayerNeuralNetwork_meanSquareErrorTrained = 0;
MultilayerNeuralNetwork_meanAbsoluteErrorTrained = 0;
MultilayerNeuralNetwork_meanSquareErrorTest = 0;
MultilayerNeuralNetwork_meanAbsolutErrorTest = 0;
for k = 1:1
    
    test_data = x((k-1)*length(x)/10+1:(k)*length(x)/10,:);
    ytest = y((k-1)*length(y)/10+1:(k)*length(y)/10,:);
    if k == 1
        train_data = x((k)*length(x)/10+1:length(x),:);
        ytrain = y((k)*length(y)/10+1:length(y),:);
    elseif k==10
       train_data = x(1:9*length(x)/10,:);
       ytrain = y(1:9*length(y)/10,:);
    else
        train_data = [x(1:(k-1)*length(x)/10,:); x(k*length(x)/10+1:length(x),:)];
        ytrain = [y(1:(k-1)*length(y)/10,:); y(k*length(y)/10+1:length(y),:)];
    end
    ytrain = num2cell(ytrain);
    ytest = num2cell(ytest);
    train_data = num2cell(train_data);
    test_data = num2cell(test_data);
    net = fitnet(10);
    net.numInputs=14;
    net = train(net,train_data',ytrain');
    %net.inputs.size = num2cell([1 ;1 ;1 ;1 ;1 ;1 ;1 ;1 ;1 ;1 ;1 ;1 ;1 ;1]);
    resultsTrained = net(train_data');
    resultsTest = net(test_data');
    net.performFcn ='mse';
    MultilayerNeuralNetwork_meanSquareErrorTrained = MultilayerNeuralNetwork_meanSquareErrorTrained + perform(net,resultsTrained,ytrain');
    MultilayerNeuralNetwork_meanSquareErrorTest = MultilayerNeuralNetwork_meanSquareErrorTest + perform(net,resultsTest,ytest');
    net.performFcn ='mae';
    MultilayerNeuralNetwork_meanAbsoluteErrorTrained = MultilayerNeuralNetwork_meanAbsoluteErrorTrained + perform(net,resultsTrained,ytrain');
    MultilayerNeuralNetwork_meanAbsolutErrorTest = MultilayerNeuralNetwork_meanAbsolutErrorTest + perform(net,resultsTest,ytest');
    
end

MultilayerNeuralNetwork_meanSquareErrorTrained =MultilayerNeuralNetwork_meanSquareErrorTrained/10
MultilayerNeuralNetwork_meanAbsoluteErrorTrained =MultilayerNeuralNetwork_meanAbsoluteErrorTrained/10
MultilayerNeuralNetwork_meanSquareErrorTest = MultilayerNeuralNetwork_meanSquareErrorTest/10
MultilayerNeuralNetwork_meanAbsolutErrorTest = MultilayerNeuralNetwork_meanAbsolutErrorTest/10
