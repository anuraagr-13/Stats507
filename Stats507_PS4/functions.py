import numpy as np
import scipy.stats as sci
import math
def normal_th(data_samp,level,str_par='Yes'):
    """
    Calculating the point estimate and confidence interval for normal
    
    Parameters
    ----------
    dataset : numpy array
    level: int
    Specifies the confidence level for the estimation
    str_par: str
    Specifies the configuration of the output
    Returns
    -------
    var: dict
    Gives the output parameters as a dictionary
    eq_str: str
    Gives the output parameters as a string
    """
    if(np.array(data_samp).ndim==1):
        var={} #Defining the dictionary
        est=np.mean(data_samp)  #point estimate
        sd_bar=np.std(data_samp)/math.sqrt(30)
        #Confidence Intervals
        if(level==90):
            z=1.645
        elif(level==95):
            z=1.96
        elif(level==99):
            z=2.576
        upper=est+(z*sd_bar)
        lower=est-(z*sd_bar)
        # Rounding the variables
        est= round(est,4)
        lower=round(lower,4)
        upper=round(upper,4)
        if(str_par==None):
            var={'level':level,'est':est,'lwr':lower,'upr':upper}
            return var
        else:
            eq_str="{0:.3f}[{1}%CI:({2:.3f},{3:.3f})]"
            eq_str=eq_str.format(est,level,lower,upper)
            return eq_str
    else:
        raise Exception("Wrong type of input")

def binom(dataset,level,method,str_par='Yes'):
    """
    Calculating the point estimate and confidence interval for normal
    
    Parameters
    ----------
    dataset : numpy array
    level: int
    Specifies the confidence level for the estimation
    method: 1
    A paramter controlling the methods
    str_par: str
    Specifies the configuration of the output
    Returns
    -------
    var: dict
    Gives the output parameters as a dictionary
    eq_str: str
    Gives the output parameters as a string
    """
    
    # Variable definitions
    var={} 
    count=0
    n_data=len(dataset)
    if(level==90):
        z=1.645
    elif(level==95):
        z=1.96
    elif(level==99):
        z=2.576
    for i in dataset:
        if(i==1):
            # Number of success in the dataset
            count=count+1
    if(np.array(dataset).ndim==1):
        #Point estimate
        est=count/n_data
        if method==1: 
        # 1. Normal Approximation
            #Confidence Intervals
            upper=est+(z*(math.sqrt(est*(1-est)/n_data)))
            lower=est-(z*(math.sqrt(est*(1-est)/n_data)))
            # Condition Check
            binom_check=(est*n_data)**(n_data*(1-est))
            if(binom_check<=12):
                print("Warning: The approximation is not adequate")
        elif(method==2):
            # 2. Clopper-Pearson Interval
            lower=sci.beta.ppf((1-(level/100))/2,count,n_data-count+1)
            upper=sci.beta.ppf(1-(1-(level/100))/2,count+1,n_data-count)
        elif(method==3):
            # 3. Jeffrey's Interval
            lower=max(0,sci.beta.ppf((1-(level/100))/2,count+0.5,n_data-count+0.5))
            upper=min(sci.beta.ppf(1-(1-(level/100))/2,count+0.5,n_data-count+0.5),1)
        else:
            # 4. Agresti–Coull interval
            n_bar=n_data+(z**2)
            p_bar=1/n_bar*(count+((z**2)/2))
            upper=p_bar+(z*(math.sqrt(p_bar*(1-p_bar)/n_bar)))
            lower=p_bar-(z*(math.sqrt(p_bar*(1-p_bar)/n_bar)))
        # Rounding the variables
        est= round(est,4)
        lower=round(lower,4)
        upper=round(upper,4)
        if(str_par==None):
            var={'level':level,'est':est,'lwr':lower,'upr':upper}
            return var
        else:
            eq_str="{0:.4f}[{1}%CI:({2:.4f},{3:.4f})]"
            eq_str=eq_str.format(est,level,lower,upper)
            return eq_str
    else:
        raise Exception("Wrong type of input")