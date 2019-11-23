**Parameters:** 14410  
**Validation Accuracy:** 99.42 in 9th epoch,99.51 in 19th epoch  
  
**Strategy:**  
1.Started with model built in assignment1- 125k params & 99.26 Vacc.  
2.Removed biases that reduced parameters slightly-124k params & 99.1 vacc.  
3.Used fewer kernels & fixed network architecture-14798 params & 98.93 Vacc  
4.used GAP -12930 params & 98.76 Vacc  
5.Added batch normalization-Reduced gap between training & validation accuracy.  
6.Added a scheduler and increased no.of epochs to 20 that gave a sharp jump in Vacc.  
7.Increased batch size.Tried 32,64,128,256 and got better results for 64.  
8.Slightly increased no.of kernels.  
9.added dropouts to avoid overfitting.Tried with uniform rates initially,but then used lower values for shallow kernels with lesser number of parameters and larger values of dropouts for deeper kernels.  
10.Achieved what was asked for :),but wondering is there an better way instead of so much trial and error and guesswork in deciding on correct no. of filters, epoch rates and other parameters:(?
