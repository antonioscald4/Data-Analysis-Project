setwd("C:/Users/anton/Downloads")
sdata <- read.csv("RegressionExam_8feb23.csv", header = TRUE, sep = ",")
sdata
library(leaps)
library(glmnet)

#predict function for regsubsets
predict.regsubsets = function(object,newdata,id,...){ # ... <-> ellipsis
  form=as.formula(object$call[[2]])
  mat=model.matrix(form, newdata)
  coefi=coef(object, id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}

######### Training and Validation Set ######## 
# create a vector that allocates each observation to one of k=5 folds and create a matrix to store the results.
k=5 # each fold has 16 samples, the 20% of the total number of samples. 
set.seed(1)
folds = sample(1:k,nrow(sdata),replace=TRUE)
cv.errors = matrix(NA,k,45, dimnames=list(NULL, paste(1:45)))

## Multiple Linear Regression ##
MLR_errors_vec = c(0,0,0,0,0) #accumulation vector for the MSEs computed in the Cross-Validation
for(j in 1:k){
  fit.MLR = lm(Y~., data=sdata[folds!=j,])
  pred = predict(fit.MLR, sdata[folds==j,], id=i)
  MLR_errors_vec[j] = mean((sdata$Y[folds==j]-pred)^2)
}
MLR_error = mean(MLR_errors_vec) # estimated test mse = 9354.128
summary(fit.MLR) # X23, X24, x25 have low p-value
coef.MLR = coef(fit.MLR)

## We do not execute Best Subset Selection since the number of predictors is high

## Forward Stepwise Selection ##
for(j in 1:k){
  regfit.fwd=regsubsets(Y~.,data=sdata[folds!=j,],nvmax=45, method ="forward")
  for(i in 1:45){
    pred = predict(regfit.fwd, sdata[folds==j,], id=i)
    cv.errors[j,i] = mean((sdata$Y[folds==j]-pred)^2)
  }
}
mean.cv.errors=colMeans(cv.errors) # mean of the columns of the errors matrix
par(mfrow=c(1,1))
dev.new()
plot(mean.cv.errors, type="b") 
coef.fwd = coef(regfit.fwd,which.min(mean.cv.errors)) # the model with X23, X24, x25 has been selected (plus the intercept)
min.cv.error.fwd = min(mean.cv.errors) # estimated test mse = 2389.453


## Backward Stepwise Selection ##
for(j in 1:k){
  regfit.bwd=regsubsets(Y~.,data=sdata[folds!=j,],nvmax=45, method ="backward")
  for(i in 1:45){
    pred = predict(regfit.bwd, sdata[folds==j,], id=i)
    cv.errors[j,i] = mean((sdata$Y[folds==j]-pred)^2)
  }
}
mean.cv.errors=colMeans(cv.errors) # mean of the columns of
par(mfrow=c(1,1))
dev.new()
plot(mean.cv.errors, type="b")
coef.bwd = coef(regfit.bwd,which.min(mean.cv.errors)) # the model with X23, X24, x25 has been selected (plus the intercept)
min.cv.error.bwd = min(mean.cv.errors) # estimated test mse =2389.453


## Hybrid Stepwise Selection ##
for(j in 1:k){
  regfit.hyb=regsubsets(Y~.,data=sdata[folds!=j,],nvmax=45, method ="seqrep")
  for(i in 1:45){
    pred = predict(regfit.hyb, sdata[folds==j,], id=i)
    cv.errors[j,i] = mean((sdata$Y[folds==j]-pred)^2)
  }
}
mean.cv.errors=colMeans(cv.errors) # mean of the columns of
par(mfrow=c(1,1))
dev.new()
plot(mean.cv.errors, type="b")
coef.hyb = coef(regfit.hyb,which.min(mean.cv.errors)) # the model with X23, X24, x25 has been selected (plus the intercept)
min.cv.error.hyb = min(mean.cv.errors) # estimated test mse = 2389.453

## Ridge Regression ##
x = model.matrix(Y~., sdata)[,-1] # original data without the Ys
y = sdata$Y
# grid=10^seq(10,-2,length=1000) #possible values of lambda from 10^10 to 10^-2
mse_ridge_vec = c(0,0,0,0,0) #accumulation vector for the MSEs computed in the Cross-Validation
for(j in 1:k){
  ridge.mod=cv.glmnet(x[folds!=j,],y[folds!=j],alpha=0,nfolds=k) #performs 5-fold cross validation 
  bestlam=ridge.mod$lambda.min
  pred = predict(ridge.mod,s=bestlam ,newx=x[folds==j,])
  mse_ridge_vec[j] <- mean((pred-y[folds==j])^2)
}
mse_ridge = mean(mse_ridge_vec) # estimated test mse = 9362426.786
coef.ridge = coef(ridge.mod) # No coefficients have value approximatetely zero
dev.new()
plot(ridge.mod,label = T, xvar = "lambda") # selected lambda = 920.885


## LASSO Regression ##
mse_lasso_vec = c(0,0,0,0,0) #accumulation vector for the MSEs computed in the Cross-Validation
for(j in 1:k){
  lasso.mod=cv.glmnet(x[folds!=j,],y[folds!=j],alpha=1,nfolds=k) #performs 5-fold cross validation 
  bestlam=lasso.mod$lambda.min
  pred = predict(lasso.mod,s=bestlam ,newx=x[folds==j,])
  mse_lasso_vec[j] <- mean((pred-y[folds==j])^2)
}
mse_lasso = mean(mse_lasso_vec) # estimated test mse = 156831.4
coef.lasso = coef(lasso.mod) # LASSO selects X23,X24,X25 (plus the intercept)
dev.new()
plot(lasso.mod,label = T, xvar = "lambda")


## Elastic Net Regression ##
mse_enet_vec = c(0,0,0,0,0) #accumulation vector for the MSEs computed in the Cross-Validation
alpha = 0.85
for(j in 1:k){
  enet.mod=cv.glmnet(x[folds!=j,],y[folds!=j],alpha=alpha,nfolds=k)
  bestlam=enet.mod$lambda.min
  pred = predict(enet.mod,s=bestlam,newx=x[folds==j,])
  mse_enet_vec[j] <- mean((pred-y[folds==j])^2)
}
mse_enet = mean(mse_enet_vec) # estimated test mse = 150796.9 (selects the same variables as LASSO, but has a smaller estimated test error)
coef.enet = coef(enet.mod)
dev.new()
plot(enet.mod,label = T, xvar = "lambda")
