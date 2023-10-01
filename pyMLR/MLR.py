import xarray as xr
import geopandas as gpd
import numpy as np
import salem
import pandas as pd
import statsmodels.api as sm
from LOTUS_regression.predictors import load_data
import LOTUS_regression.predictors as predictors
import scipy.stats
from dominance_analysis import Dominance_Datasets
from dominance_analysis import Dominance
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


def norm(a,time):
    a = a.sel(time=time)
    return (a-a.mean())/a.std()
def norm_std_1D(path,time):
    a = xr.open_dataarray(path).sel(time=time)
    return (a-a.mean())/a.std()
def norm_1D(path,time):
    a = xr.open_dataarray(path).sel(time=time)
    return (a-a.mean())
def lotus_pred(variable,time):
    predictors_pwlt = load_data('pred_baseline_pwlt.csv')
    return xr.DataArray(predictors_pwlt[''+str(variable)+'']).sel(time=time)
def norm_std_2D(path,shapefile_xarray,time,dims):
    a = xr.open_dataarray(path).sel(time=time).salem.roi(shape=shapefile_xarray).mean(dim=dims).interpolate_na(dim='time')
    return (a-a.mean())/a.std()
def norm_2D(path,shapefile_xarray,time,dims):
    a = xr.open_dataarray(path).sel(time=time).salem.roi(shape=shapefile_xarray).mean(dim=dims).interpolate_na(dim='time')
    return (a-a.mean())
def norm_std_2D(path,shapefile_xarray,time,dims):
    a = xr.open_dataarray(path).sel(time=time).salem.roi(shape=shapefile_xarray).mean(dim=dims).interpolate_na(dim='time')
    return (a-a.mean()/a.std())

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
    bb,cc,dd = pd.DataFrame(aa[1]), pd.DataFrame(aa[4]), pd.DataFrame(aa[2])
    bb.to_csv('test_mlr_corr_coefs.csv', header=None)
    cc.to_csv('test_mlr_p_values.csv', header=None)
    dd.to_csv('test_mlr_stderr.csv', header=None)
    kk = pd.read_csv('test_mlr_corr_coefs.csv', header=None)
    pd.DataFrame((res.tables[0][0])).to_csv('test_mlr_r2.csv')
    pd.DataFrame((res.tables[0][1])).to_csv('test_mlr_adj_r2.csv')
    r = pd.read_csv('test_mlr_r2.csv', header=None)
    adj_r = pd.read_csv('test_mlr_adj_r2.csv', header=None)
    p_val = pd.read_csv('test_mlr_p_values.csv', header=None)
    stderr = pd.read_csv('test_mlr_stderr.csv', header=None)
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
    p_values, p_val_cons, stderr_v, stderr_co = np.array(p_val[1][:-1][::-1]), p_val[1][len(p_val[1])-1], np.array(stderr[1][:-1][::-1]), stderr[1][len(p_val[1])-1]
    return model, np.array(coeffs), cons,  r_sq, adj_rsq, res, p_values, p_val_cons, stderr_v, stderr_co, print('\nR_squared = '+str(r_sq)+'\nadj_Rsq = '+str(adj_rsq)+'\n "om!"')


def mlr_contributions(y,mlr_model,v0):
    k, kk = [],[]
    for i in range(0,len(v0)):
        for j in range(0,len(v0)):
            if i!=j:
                k.append(v0[j])
            else:
                pass
        kk.append(k)
        k=[]
    cont = []
    for h in range(len(kk)):
        mlr_model1 = mlr_out(y,kk[h])
        cont.append(((mlr_model1[6].ssr-mlr_model[6].ssr)/mlr_model1[6].ssr)*100)
    return cont

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

#Seasonal

def time_series_2D(path, shapefile_xarray, time, dims):
    try:
        return xr.open_dataarray(path).sel(time=time).salem.roi(shape=shapefile_xarray).mean(dim=dims).interpolate_na(dim='time')
    except:
        return xr.open_dataarray(path).sel(time=time).mean(dim=dims).interpolate_na(dim='time')
def time_series_1D(path, time):
    return xr.open_dataarray(path).sel(time=time).interpolate_na(dim='time')

def djf(y,t1,t2): 
    mons = [12,1,2]
    y = y.sel(time=slice(''+str(t1)+'',''+str(t2)+''))
    y1 = y.sel(time=y.time.dt.month.isin([mons]))[2:]
    n = len(mons)
    nn = int(len(y1)/n)
    y_new = y1[0:(nn*n)]
    if y_new.ndim!=1:
        oi = []
        for i1 in range(len(y_new.lev)):
            xa = y_new.isel(lev=i1)
            xa1 = np.average(np.array(xa).reshape(-1, 3), axis=1)
            oi.append(xa1)
        data = xr.DataArray(oi, coords=[y_new.lev, pd.date_range('01-01-'+str(t1)+'','01-01-'+str(int(t2)-1)+'',freq='YS')], dims=['lev','time'])
        oi=[]
    else:
        y1 = np.average(np.array(y_new).reshape(-1, 3), axis=1)
        data = xr.DataArray(y1, coords=[pd.date_range('01-01-'+str(t1)+'','01-01-'+str(int(t2)-1)+'',freq='YS')], dims=['time'])
    return data




def mam(y, t1,t2):
    mons = [3,4,5]
    y1 = y.sel(time=y.time.dt.month.isin([mons])).resample(time='YS').mean(dim='time').sel(time=slice(''+str(t1)+'',''+str(t2)+''))
    return y1

def jjas(y, t1,t2):
    mons = [6,7,8,9]
    y1 = y.sel(time=y.time.dt.month.isin([mons])).resample(time='YS').mean(dim='time').sel(time=slice(''+str(t1)+'',''+str(t2)+''))
    return y1

def jja(y, t1,t2):
    mons = [6,7,8]
    y1 = y.sel(time=y.time.dt.month.isin([mons])).resample(time='YS').mean(dim='time').sel(time=slice(''+str(t1)+'',''+str(t2)+''))
    return y1

def on(y, t1,t2):
    mons = [10,11]
    y1 = y.sel(time=y.time.dt.month.isin([mons])).resample(time='YS').mean(dim='time').sel(time=slice(''+str(t1)+'',''+str(t2)+''))
    return y1

def son(y, t1,t2):
    mons = [9,10,11]
    y1 = y.sel(time=y.time.dt.month.isin([mons])).resample(time='YS').mean(dim='time').sel(time=slice(''+str(t1)+'',''+str(t2)+''))
    return y1


##############Multicollinearity

def correlation_matrix(var,headss,path_and_filename_to_save_the_output,fs,figsize): ###var is the list of predictors, heads the title of predictors
    df = pd.DataFrame(pd.DataFrame(np.array(var)).T)
    df.columns=headss
    matrix = df.corr()
    fig, ax = plt.subplots(1,1,figsize=figsize, dpi=100)
    cs = ax.matshow(matrix, cmap='RdYlGn_r', vmin=-1, vmax=1)
    ax.set_xticks(range(0,len(headss)))
    ax.set_yticks(range(0,len(headss)))
    ax.set_xticklabels(headss, fontsize=fs)
    ax.set_yticklabels(headss, fontsize=fs)
    cax = plt.axes([0.92,0.18,0.04,0.65])
    for (i, j), z in np.ndenumerate(matrix):
        ss = '{:0.2f}'.format(z)
        if ss != 'nan':
            ax.text(j, i, ss, ha='center', va='center', fontsize=fs,
                    bbox=dict(boxstyle='round', facecolor='white',alpha=0.75, edgecolor='black',linewidth=0.2), )
    cbar = plt.colorbar(cs, cax=cax)
    cbar.ax.tick_params(labelsize=fs)
    fig.savefig(''+str(path_and_filename_to_save_the_output)+'', dpi=500, bbox_inches='tight', facecolor='white')
    plt.show()



def VIF(var, headss):
    X = pd.DataFrame(pd.DataFrame(np.array(var)).T)
    X.columns=headss
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                              for i in range(len(X.columns))]
    print(vif_data)



#EXAMPLE FOR USING THE FUNCTION


