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

## Training and Test Set
n=64
train = sdata[row.names(sdata) %in% 1:n, ]
test = sdata[row.names(sdata) %in% (n+1):nrow(sdata), ]

######### Training and Validation Set ######## 
# create a vector that allocates each observation to one of k=8 folds and create a matrix to store the results.
k=8 # each fold has 16 samples, the 20% of the total number of samples. 
set.seed(1)
folds = sample(1:k,nrow(train),replace=TRUE)
cv.errors = matrix(NA,k,45, dimnames=list(NULL, paste(1:45)))

## Multiple Linear Regression ##
# we perform it only as preliminary step, since in this setting this method has high variance
MLR_errors_vec = c(0,0,0,0,0,0,0,0) #accumulation vector for the MSEs computed in the Cross-Validation
for(j in 1:k){
  fit.MLR = lm(Y~., data=train[folds!=j,])
  pred = predict(fit.MLR, train[folds==j,], id=i)
  MLR_errors_vec[j] = mean((train$Y[folds==j]-pred)^2)
}
MLR_error = mean(MLR_errors_vec) # estimated test mse = 22266.85
summary(fit.MLR) # X23, X24, x25 have low p-value
coef.MLR = coef(fit.MLR)
test_error_mlr = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
for (i in 1:16){
  pred = predict(fit.MLR, test, id=i)
  test_error_mlr[i] = mean((test$Y-pred)^2) 
}
test_error_mlr_tot = mean(test_error_mlr) # test error 7602.428

## We do not execute Best Subset Selection since the number of predictors is high

## Forward Stepwise Selection ##
for(j in 1:k){
  regfit.fwd=regsubsets(Y~.,data=train[folds!=j,],nvmax=45, method ="forward")
  for(i in 1:45){
    pred = predict(regfit.fwd, train[folds==j,], id=i)
    cv.errors[j,i] = mean((train$Y[folds==j]-pred)^2)
  }
}
mean.cv.errors=colMeans(cv.errors) # mean of the columns of the errors matrix
par(mfrow=c(1,1))
dev.new()
plot(mean.cv.errors, type="b") 
coef.fwd = coef(regfit.fwd,which.min(mean.cv.errors)) # the model with X23, X24, x25 has been selected (plus the intercept)
min.cv.error.fwd = min(mean.cv.errors) # estimated test mse = 3128.131
test_error_fwd = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
for (i in 1:16){
  pred = predict(regfit.fwd, test, id=i)
  test_error_fwd[i] = mean((test$Y-pred)^2) 
}
test_error_fwd_tot = mean(test_error_fwd) # test error 8134904


## Backward Stepwise Selection ##
for(j in 1:k){
  regfit.bwd=regsubsets(Y~.,data=train[folds!=j,],nvmax=45, method ="backward")
  for(i in 1:45){
    pred = predict(regfit.bwd, train[folds==j,], id=i)
    cv.errors[j,i] = mean((train$Y[folds==j]-pred)^2)
  }
}
mean.cv.errors=colMeans(cv.errors) # mean of the columns of
par(mfrow=c(1,1))
dev.new()
plot(mean.cv.errors, type="b")
coef.bwd = coef(regfit.bwd,which.min(mean.cv.errors)) # the model with X23, X24, x25 has been selected (plus the intercept)
min.cv.error.bwd = min(mean.cv.errors) # estimated test mse = 3128.131
test_error_bwd = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
for (i in 1:16){
  pred = predict(regfit.bwd, test, id=i)
  test_error_bwd[i] = mean((test$Y-pred)^2) 
}
test_error_bwd_tot = mean(test_error_bwd) # test error 8135297

## Hybrid Stepwise Selection ##
for(j in 1:k){
  regfit.hyb=regsubsets(Y~.,data=train[folds!=j,],nvmax=45, method ="seqrep")
  for(i in 1:45){
    pred = predict(regfit.hyb, train[folds==j,], id=i)
    cv.errors[j,i] = mean((train$Y[folds==j]-pred)^2)
  }
}
mean.cv.errors=colMeans(cv.errors) # mean of the columns of
par(mfrow=c(1,1))
dev.new()
plot(mean.cv.errors, type="b")
coef.hyb = coef(regfit.hyb,which.min(mean.cv.errors)) # the model with X23, X24, x25 has been selected (plus the intercept)
min.cv.error.hyb = min(mean.cv.errors) # estimated test mse = 3128.131
test_error_hyb = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
for (i in 1:16){
  pred = predict(regfit.hyb, test, id=i)
  test_error_hyb[i] = mean((test$Y-pred)^2) 
}
test_error_hyb_tot = mean(test_error_hyb) # test error 24120551

## Ridge Regression ##
x = model.matrix(Y~., train)[,-1] # original data without the Ys
y = train$Y
# grid=10^seq(10,-2,length=1000) #possible values of lambda from 10^10 to 10^-2
mse_ridge_vec = c(0,0,0,0,0,0,0,0) #accumulation vector for the MSEs computed in the Cross-Validation
for(j in 1:k){
  ridge.mod=cv.glmnet(x[folds!=j,],y[folds!=j],alpha=0,nfolds=k) #performs 8-fold cross validation 
  bestlam=ridge.mod$lambda.min
  pred = predict(ridge.mod,s=bestlam ,newx=x[folds==j,])
  mse_ridge_vec[j] <- mean((pred-y[folds==j])^2)
}
mse_ridge = mean(mse_ridge_vec) # estimated test mse = 8206254
coef.ridge = coef(ridge.mod) # No coefficients have value approximatetely zero
dev.new()
plot(ridge.mod,label = T, xvar = "lambda") # selected lambda = 922.6
x_test = model.matrix(Y~., test)[,-1] # original data without the Ys
y_test = test$Y
test_error_ridge = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
for (i in 1:16){
  pred = predict(ridge.mod, x_test, id=i)
  test_error_ridge[i] = mean((test$Y-pred)^2) 
}
test_error_ridge_tot = mean(test_error_ridge) # test error 7007760

## LASSO Regression ##
mse_lasso_vec = c(0,0,0,0,0,0,0,0) #accumulation vector for the MSEs computed in the Cross-Validation
for(j in 1:k){
  lasso.mod=cv.glmnet(x[folds!=j,],y[folds!=j],alpha=1,nfolds=k) #performs 8-fold cross validation 
  bestlam=lasso.mod$lambda.min
  pred = predict(lasso.mod,s=bestlam ,newx=x[folds==j,])
  mse_lasso_vec[j] <- mean((pred-y[folds==j])^2)
}
mse_lasso = mean(mse_lasso_vec) # estimated test mse = 159072.3
coef.lasso = coef(lasso.mod) # LASSO selects X23,X24,X25 (plus the intercept) 
dev.new()
plot(lasso.mod,label = T, xvar = "lambda")
x_test = model.matrix(Y~., test)[,-1] # original data without the Ys
y_test = test$Y
test_error_lasso = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
for (i in 1:16){
  pred = predict(lasso.mod, x_test, id=i)
  test_error_lasso[i] = mean((test$Y-pred)^2) 
}
test_error_lasso_tot = mean(test_error_lasso) # test error 140368.2


## Elastic Net Regression ##
mse_enet_vec = c(0,0,0,0,0,0,0,0) #accumulation vector for the MSEs computed in the Cross-Validation
alpha = 0.90
for(j in 1:k){
  enet.mod=cv.glmnet(x[folds!=j,],y[folds!=j],alpha=alpha,nfolds=k)
  bestlam=enet.mod$lambda.min
  pred = predict(enet.mod,s=bestlam,newx=x[folds==j,])
  mse_enet_vec[j] <- mean((pred-y[folds==j])^2)
}
mse_enet = mean(mse_enet_vec) # estimated test mse = 151911 (selects the same variables as LASSO, but has a smaller estimated test error)
coef.enet = coef(enet.mod) # Elastic Net selects X23,X24,X25 (plus the intercept) 
dev.new()
plot(enet.mod,label = T, xvar = "lambda")
x_test = model.matrix(Y~., test)[,-1] # original data without the Ys
y_test = test$Y
test_error_enet = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
for (i in 1:16){
  pred = predict(enet.mod, x_test, id=i)
  test_error_enet[i] = mean((test$Y-pred)^2) 
}
test_error_enet_tot = mean(test_error_enet) # test error 158178.3


