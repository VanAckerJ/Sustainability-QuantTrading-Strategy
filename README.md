# import relevant python libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# import data frame from dropbox file we created:
# this file contains return data for the replicating portfolio

url = "https://dl.dropboxusercontent.com/s/n9lpnam1d81jwwh/Copy%20of%20ETF%20Replication%201.xlsx?dl=0"
data = pd.read_excel(url, index_col=0,sheet_name=6, na_values=-99)
data.index = pd.to_datetime(data.index,format='%m/%d/%Y').strftime('%m/%d/%Y')
data.head(2)

# Import the factors (RF, MKT, SMB, HML, RWC) 

Factors = pd.read_excel(url, index_col=0, sheet_name=3, na_values=-99)
Factors = Factors/100
Factors.index = pd.to_datetime(Factors.index, format='%m/%d/%Y').strftime('%m/%d/%Y')
Factors.head()


# Merging the two dataframes together so all data is in a single dataframe --> allows simple manipulation
# The replicated portfolio column represents past returns. 
# These returns have been implemented with the most recent weight configuration listed by Ishares interactive prospectus, 
# so each asset has been multiplied by its corresponding weight. (See Excel file: sheet 5 for implemented weights)

data = pd.merge(data, Factors, right_index=True, left_index=True)
data.head(5)


# Important note about RMW (profitability factor): 
# The RMW factor represents the difference between the returns on diversified portfolios of stocks with robust and weak 
# profitability; i.e. robust profitability in excess of weak profitability 
# All variables provide different combinations of exposures to the unknown state variables.


# Used excess returns to better test our strategy   

data.iloc[:,1]=data.iloc[:,1].subtract(data['RF'],axis=0)
data.iloc[:,2]=data.iloc[:,2].subtract(data['RF'],axis=0)
data.head(5)

# Remove the risk-free rate from the Dataframe: 

del data['RF']
data.head(2)

# Four Factor Model analysis: 

# Estimate time series regression for the replicating portfolio   

# import stats package to run OLS multivariate regression:
import statsmodels.api as sm
fourFts=pd.DataFrame([],index=data.drop(['MKT','HML','SMB','RMW'],axis=1).columns,\
                 columns=['avg','alpha','talpha','betamkt','tbetamkt','betahml','tbetahml','betasmb','tbetasmb','betarmw',\
                          'tbetarmw'])

# create dataframe to store residuals:
fourFtsResid=data.drop(['MKT','HML','SMB','RMW'],axis=1).copy()

# create new column (avg) and then store avg returns in that column:
fourFts['avg']=data.drop(['MKT','HML','SMB','RMW'],axis=1).mean()

# run for loop through the four factors and store all data accordingly:
for portfolio in fourFts.index:
    y=data[portfolio]
    x=data[['MKT','HML','SMB','RMW']]
    x=sm.add_constant(x)
    results = sm.OLS(y,x).fit()
    fourFts.at[portfolio,['alpha','betamkt','betahml','betasmb','betarmw']]=results.params.values
    fourFts.at[portfolio,['talpha','tbetamkt','tbetahml','tbetasmb','tbetarmw']]=(results.params/results.HC0_se).values
    fourFtsResid[portfolio]=results.resid

# storing regression as a float:    
fourFts=fourFts.astype('float')    
fourFts

# Beta and Talpha interpretation

# Given the regression output above, we noticed that the "betarmw" (and all betas for that matter, aside from "betasmb"), 
# is negative. This means that our replicating portfolio does not comove with RMW.  


# Results from the Four Factor Regression:
results.summary()

# Low R-squared --> indicates that changes in the predictors are quite unrelated to changes in the response variable and 
# that our four factor model explains very little of the response variability. This low r-squared is somewhat  
# problematic considering we are trying to formulate accurate predictions about the model. 

# low const (alpha) --> not a lot of variation in returns not explained by the four factor model
# high const (alpha) T Stat --> This value is statistically significant at the 99% confidence level. 
# The t stat tests the hypothesis that the true value of the coefficient is non-zero in order to confirm that independent 
# variables truly belong in our model. 

# four factor low T Stats -->  i.e. are statistically insignificant thus the measure of precision of which the regression
# coefficient is measured is insignificant. Considering we did not form portfolios based on size and value, such as portfolios
# with high/low BE/ME or size differences, we did not expect to see statistically significant t stats for the corresponding
# factors SMB and HML. 

# We then performed a multivariate GRS test and reported the GRS F-statistic 
muFourF=data[['MKT','HML','SMB','RMW']].mean()
invCovFourF=np.linalg.inv(data[['MKT','HML','SMB','RMW']].cov())
T=data.shape[0]
# the inverse of the residual covariance matrix
invCov=np.matrix(np.linalg.inv(fourFtsResid.cov()))
# chi statistic below
chi=T*(1+(muFourF.values @ invCovFourF @ muFourF.values.T))**(-1)*(fourFts.alpha.values @invCov@ fourFts.alpha.values.T)
# F and chi-squared statistics are really the same thing in that, after a normalization, chi-squared is the limiting distribution 
# of the F as the denominator degrees of freedom goes to infinity
chi



# This reports the p value given the GRS test for the four-factor model 
from scipy.stats import chi2
dfreedom=fourFtsResid.shape[1]

(1-chi2.cdf(chi,dfreedom))*100

# Given this p-value of ~0.009, we can reject the null that the coefficients are different from 0 at the 99% confidence level.
# This p value corresponds to the probability of observing such an extreme p-value as .00924199 essentially by chance.
# The low R-Squared 

# Here, we created a new dataframe to include just the (Rm-Rf) and our replicated portfolio 
Rep_MKT = data[['Replicated_Portfolio','MKT']]

# Next, we took a look at the CAPM alphas relative to the 3 factor model by running a time series regression  
CAPM=pd.DataFrame([],index=Rep_MKT.drop(['MKT'],axis=1).columns,columns=['avg','alpha','talpha','beta','tbeta'])   
CAPM['avg']=Rep_MKT.drop(['MKT'],axis=1).mean()

for portfolios in CAPM.index:
    y=Rep_MKT[portfolios]
    x=Rep_MKT['MKT']
    x=sm.add_constant(x)
    results1 = sm.OLS(y,x).fit()
    CAPM.at[portfolios,'alpha']=results1.params[0]
    CAPM.at[portfolios,'talpha']=results1.params[0]/results1.HC0_se[0]
    CAPM.at[portfolios,'beta']=results1.params[1]
    CAPM.at[portfolios,'tbeta']=results1.params[1]/results1.HC0_se[1]
CAPM


# This cell represents the summary statistics of the CAPM regression. The r-squared relative to the four-factor model's is 
# much lower, indicating that changes in the predictors are quite unrelated to changes in the response variable and 
# that our CAPM model explains very little of the response variability. Given that R-squared represents the "scatter" 
# around the regression line, this low R-squared indicates high variation and scatter around the regression line.  

# However, the t statistic of 3.733 is indicative of something. The ratio of the sample regression coefficient to its standard 
# error is the t-statistic, and in our CAPM regression. This t-stat can be viewed as being statistically significant. 
# The alpha t-stat test is the best gage for validity in our model. 

# Ultimately, we believe adding more variables to the model could better justify having significant predictors but a low 
# R-squared. It is definitely possible that additional predictors could increase the true explanatory power of our model. 
results1.summary()

# In this cell, we conducted a GRS test for the CAPM. This allowed us to find out what the F-statistic and p-value are in 
# relation to the CAPM. 

from scipy.stats import chi2

def GRS(Rep_MKT,beg,end):
    df_CAPM=Rep_MKT.copy()
    Resid_CAPM=Rep_MKT.copy()
    x= sm.add_constant(df_CAPM['MKT'])
    del Resid_CAPM['MKT']
    E_CAPM=pd.DataFrame(Resid_CAPM.mean())
    E_CAPM.columns=['alpha']
    for portfolios in E_CAPM.index:
        y= df_CAPM[portfolios]
        results_CAPM1= sm.OLS(y,x).fit()
        E_CAPM.at[portfolios,'alpha']=results_CAPM1.params[0]
        Resid_CAPM.loc[:,portfolios]=results_CAPM1.resid

    avgMkt=Rep_MKT.MKT.mean()
    stdMkt=Rep_MKT.MKT.std()
    T=Rep_MKT.shape[0]
    # the inverse of the residual covariance matrix
    invCovCAPM1=np.linalg.inv(Resid_CAPM.cov())
    chi_CAPM=T*(1+(avgMkt/stdMkt)**2)**(-1)*(E_CAPM.alpha.values @invCovCAPM1@ E_CAPM.alpha.values.T)
    N_CAPM= Resid_CAPM.shape[1]
    return [chi_CAPM,(1-chi2.cdf(chi_CAPM,N_CAPM))*100]
chi_CAPM=GRS(Rep_MKT,'2014','2018')
chi_CAPM

# At the bottom, you can view the absolute pricing erros as the sum of the absolute values of the corresponding alphas 
[CAPM.alpha.abs().sum(),fourFts.alpha.abs().sum()]

# These values indicate that the CAPM does a better job of explaining the mispricing errors. Absolute pricing errors are
# marginally smaller for the CAPM relative to the four-factor model. Considering our replicating portfolio holds several
# securities in the underlying asset, i.e. the market index, we expected to see some similarities between our the four-factor
# model's absolute pricing errors and the CAPMs.

# The four-factor absolute pricing errors being greater than the CAPMs could be interpretted several ways. Ken French discovered
# that there is what is known as an "artificial correlation" effect in play when considering correlations between HML and 
# profitability. Holding a particularly lesser position in certain assets in our replicated portfolio, due mainly to high return
# variances, could result in some miscalculations for the four factor model. 

# Sharpe ratio of our replicated portfolio 
mu_rep = data.Replicated_Portfolio.mean()*12
std_rep = data.Replicated_Portfolio.std()
SR_repl = mu_rep/(std_rep*12**0.5)
SR_repl


# Sharpe Ratio of the Market 
mu_mkt = data.MKT.mean()*12
std_mkt = data.MKT.std()
SR_mkt = mu_mkt/(std_mkt*12**0.5)
SR_mkt

# Benefits of our replicated portfolio
SR_diff = SR_repl-SR_mkt
SR_diff

# As seen from the cells above, we confirmed that implementing our trading strategy of profitability, thus replicating
# an ETF with sustainability/profitability characterstics, produces a Sharpe ratio 0.37589 higher than the market. This is 
# an indication that our replicated portfolio is more efficient relative to the market on a risk-adjusted basis.  

# Now, we figured a good way to test our strategy better was to see if by running a two factor model, we could capture
# profitability patterns in our average stock returns; specifically our replicated portfolio's returns. 

# Create a dataframe with just the replicated portfolio and the two factors: (Rm - Rf) and profitability
Re = data[['Replicated_Portfolio','MKT','RMW']]
Re

# Estimate the time series regression for the replicating portfolio in regards to a two-factor model; thus using the
# market risk premium and profitability factors as our regressors 

# import stats package to run OLS regression:
import statsmodels.api as sm
two_F=pd.DataFrame([],index=Re.drop(['MKT','RMW'],axis=1).columns,\
                 columns=['avg','alpha','talpha','betamkt','tbetamkt','betarmw','tbetarmw'])

# create dataframe to store residuals:
two_FResid=Re.drop(['MKT','RMW'],axis=1).copy()

# create new column (avg) and then store avg returns in that column:
two_F['avg']=Re.drop(['MKT','RMW'],axis=1).mean()

# run for loop through the four factors and store all data accordingly:
for portfolio in two_F.index:
    y=Re[portfolio]
    x=Re[['MKT','RMW']]
    x=sm.add_constant(x)
    results_2 = sm.OLS(y,x).fit()
    two_F.at[portfolio,['alpha','betamkt','betarmw']]=results_2.params.values
    two_F.at[portfolio,['talpha','tbetamkt','tbetarmw']]=(results_2.params/results_2.HC0_se).values
    two_FResid[portfolio]=results_2.resid

# storing regression as a float:    
two_F=two_F.astype('float')    
two_F.iloc[:,0:7]


# We then performed a multivariate GRS test and reported the GRS F-statistic given the two factor model  
mu2=Re[['MKT','RMW']].mean()
invCov2F=np.linalg.inv(Re[['MKT','RMW']].cov())
T=Re.shape[0]
# the inverse of the residual covariance matrix
invCov2=np.matrix(np.linalg.inv(two_FResid.cov()))
# chi statistic below
chi_2=T*(1+(mu2.values @ invCov2F @ mu2.values.T))**(-1)*(two_F.alpha.values @invCov2@ two_F.alpha.values.T)
# F and chi-squared statistics are really the same thing in that, after a normalization, chi-squared is the limiting distribution 
# of the F as the denominator degrees of freedom goes to infinity
chi_2

# This reports the p value given the GRS test for the two-factor model 
from scipy.stats import chi2
dfreedom_2F=two_FResid.shape[1]

(1-chi2.cdf(chi_2,dfreedom_2F))*100

# Conclusion about the Two-Factor model:

# A lower R-squared result somewhat reduces the power of the GRS test, but the p-values in the test are 0.014. Despite 
# strong rejections on the GRS test, smaller absolute intercepts relative to the four-factor model
# and relatively low estimates of the proportion of the cross-section variation of expected returns not explained 
# suggests the proficiency of the two-factor model over the four-factor model. 

# Important Note:
# Again, we noticed similarites with the CAPM. The statistical signifance of the t-stat for the CAPM relative to the two-
# factor model's "const" t-stat are practically the same. This reinforces our implications about the similarities between 
# our replicated portfolio and the underlying index, and how they perform very similarly. 

# Below represents the Appraisal ratios 
# two-factor model Appraisal ratio
ARatio_2F = two_F.alpha/two_FResid.std()
ARatio_2F['Replicated_Portfolio']
# four-factor model Appraisal ratio
ARatio_4F = fourFts.alpha/fourFtsResid.std()
ARatio_4F['Replicated_Portfolio']
# CAPM Appraisal ratio 
ARatio_CAPM = CAPM.alpha/Resid_CAPM.std()
ARatio_CAPM['Replicated_Portfolio']

# As seen in the results below, the CAPM has the lowest Appraisal ratio compared to the two and four factor model's
# The Appraisal ratio can be used to determine a manager's investment-picking ability, and the higher the appraisal ratio,
# the better the ability to efficiently "pick" investments. Comparing the replicating portfolio's alpha to the portfolio's
# unsystematic risk (residual standard deviation), we can determine the quality differentiation of the selections.   

# Sharpe ratio of tangency portfolio spanned by the four-factors
SR_Four = (muFourF.values @ invCovFourF @ muFourF.values.T)**0.5
SR_Four
# Sharpe ratio of tangency portfolio spanned by the two factors.
SR_Two = (mu2.values @ invCov2F @ mu2.values.T)**0.5
SR_Two

# At the bottom, you can view the absolute pricing erros as the sum of the absolute values of all the corresponding alphas 
[CAPM.alpha.abs().sum(),fourFts.alpha.abs().sum(), two_F.alpha.abs().sum()]

# Ultimately, our goal with the tests, regressions, and overall analysis computed above was not to prove profitability as
# existing, but rather to show how utilizing a strategy of portfolio allocation based upon past firms with sustainable 
# profitability attributes can work to generate some alpha. Having exposure to socially responsible firms with positive
# environmental, social, and governance characteristics 
