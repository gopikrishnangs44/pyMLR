import xarray as xr
import geopandas as gpd
import numpy as np
import salem
import pandas as pd
import statsmodels.api as sm
import scipy.stats
from dominance_analysis import Dominance_Datasets
from dominance_analysis import Dominance
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


def norm_1D(path,time):
    a = xr.open_dataarray(path).sel(time=time)
    return (a-a.mean())/a.std()
def lotus_pred(variable,time):
    predictors_pwlt = load_data('pred_baseline_pwlt.csv')
    return xr.DataArray(predictors_pwlt[''+str(variable)+'']).sel(time=time)
def norm_2D(path,shapefile_xarray,time,dims):
    a = xr.open_dataarray(path).sel(time=time).salem.roi(shape=shapefile_xarray).mean(dim=dims).interpolate_na(dim='time')
    return (a-a.mean())/a.std()
def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

### mlr_out returns model, coefficiants, constant ,r_sq and adj_rsq 
def mlr_out(norm_y,norm_vars):
    res = reg_m(np.array(norm_y),norm_vars).summary()
    aa = pd.DataFrame(res.tables[1][1:])
    bb = pd.DataFrame(aa[1])
    bb.to_csv('test_mlr_corr_coefs.csv', header=None)
    kk = pd.read_csv('test_mlr_corr_coefs.csv', header=None)
    pd.DataFrame((res.tables[0][0])).to_csv('test_mlr_r2.csv')
    pd.DataFrame((res.tables[0][1])).to_csv('test_mlr_adj_r2.csv')
    r = pd.read_csv('test_mlr_r2.csv', header=None)
    adj_r = pd.read_csv('test_mlr_adj_r2.csv', header=None)
    cons = kk[1][len(kk[1])-1]
    coeffs = kk[1][:-1]
    coeffs = coeffs[::-1]
    mo = []
    for i,j in zip(norm_vars,coeffs):
        dd = i*j
        mo.append(dd)
    mo1 = np.nansum(mo, axis=0)
    model = cons + mo1
    r_sq = float(r[1][4][3:])
    adj_rsq = float(adj_r[1][4][3:])
    return model, np.array(coeffs), cons,  r_sq, adj_rsq, res , print('\nR_squared = '+str(r_sq)+'\nadj_Rsq = '+str(adj_rsq)+'\n "om!"')


def mlr_dominance(dependent_var,predictors_list,list_of_header_names,mlr_model):
    varss1 = []
    varss = predictors_list
    for i in varss:
        varss1.append(np.array(i))
    varss2 = pd.DataFrame(varss1)
    aa = pd.DataFrame(varss2.T)
    aa.columns= list_of_header_names
    aa.insert(len(varss), "target", dependent_var, True)
    dominance_regression=Dominance(data=aa,target='target',objective=1)
    ee = dominance_regression.incremental_rsquare()
    cont = []
    for i in ee:
        cont.append((ee[i]/mlr_model[3])*100)
    return cont

##############Multicollinearity

def correlation_matrix(var,headss,path_and_filename_to_save_the_output,fs,figsize): ###var is the list of predictors, heads the title of predictors
    df = pd.DataFrame(pd.DataFrame(np.array(var)).T)
    df.columns=headss
    matrix = df.corr()
    fig, ax = plt.subplots(1,1,figsize=figsize, dpi=100)
    cs = ax.matshow(matrix, cmap='RdYlGn_r', vmin=-1, vmax=1)
    ax.set_xticks(range(0,len(headss)))
    ax.set_yticks(range(0,len(headss)))
    ax.set_xticklabels(headss)
    ax.set_yticklabels(headss)
    cax = plt.axes([0.92,0.18,0.02,0.65])
    for (i, j), z in np.ndenumerate(matrix):
        ss = '{:0.2f}'.format(z)
        if ss != 'nan':
            ax.text(j, i, ss, ha='center', va='center', fontsize=fs,
                    bbox=dict(boxstyle='round', facecolor='white',alpha=0.75, edgecolor='black',linewidth=0.2), )
    plt.colorbar(cs, cax=cax)
    fig.savefig(''+str(path_and_filename)+'', dpi=500, bbox_inches='tight', facecolor='white')
    plt.show()



def VIF(var, headss):
    X = pd.DataFrame(pd.DataFrame(np.array(var)).T)
    X.columns=headss
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                              for i in range(len(X.columns))]
    print(vif_data)






