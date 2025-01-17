import pandas as pd
import numpy as np
import pickle
from operator import itemgetter
import seaborn as sns
import matplotlib.pyplot as plt

from states import states
from map_maker import map_maker 
from make_districts import district_solver
from utils import make_folder
import matplotlib as mpl
sns.set_style("white")
mpl.rc('font',family='serif')


def make_barplot(df_list, state, labels):	
	"""
	Takes table output from make_district_df() and plots it. This allows us to 
	have compact results even when there is a large number of districts. 

	INPUTS:
	----------------------------------------------------------------------------
	state: string, key of "states" dict, which pairs states with abbreviations
	alphaW_list: list, list of alphaW parameters (scalars in [0,1])
	file_end: ending of filename for figure	

	OUTPUTS:
	----------------------------------------------------------------------------
	None

	"""	
	sns.set_style(style='darkgrid')
	mpl.rc('font',family='serif')
	tpf.make_folder('../analysis/' + state)

	n_subplots = len(df_list)
	cmap = mpl.cm.get_cmap('brg')
	fig, ax = sns.plt.subplots(n_subplots, 1, figsize=(7, 5), sharex=True)
	
	for idf, df in enumerate(df_list):
		df.sort_values(by='REP_PCT', inplace=True)
		colors = [cmap(i/2) for i in df['REP_PCT'].values]
		
		x1 = np.arange(len(df))
		x2 = np.arange(len(df) + 1)
		y1 = np.ones(len(df),)
		y2 = np.ones(len(df) + 1,)
		
		dem_bar = df['DEM_PCT'].values/100
		ax[idf].bar(x1, y1, color='r', linewidth=0, width=1.0, alpha=.8)
		ax[idf].bar(x1, dem_bar, color='b', linewidth=0, width=1.0, alpha=.8)
		
		# horizontile line at .5
		ax[idf].plot(x2, y2*.5, color='w', linewidth=.2, alpha=.8)

		ax[idf].set_xticklabels([])
		ax[idf].set_xlim(0, len(df))
		ax[idf].set_ylim(0, 1)
		ax[idf].set_ylabel(labels[idf])		

	filename = '../analysis/' + state + '/barplot.pdf'
	fig.savefig(filename, bbox_inches='tight', dpi=300)
	plt.close()

	return None



def calculate_EG(df_list):
#df_list comes from make_district_df()
#Assumption at the moment: all the districts have equal population.

        output = []

        for df in df_list:
                wasted_Rep = 0
                wasted_Dem = 0
                pct_vec = df['REP_PCT']/100
                for x in pct_vec:

                        if x < .5:
                                wasted_Rep += x/len(df)
                                wasted_Dem += (.5 - x)/len(df)

                        if x > .5:
                                wasted_Rep += (x-.5)/len(df)
                                wasted_Dem +=(1-x)/len(df)

                output.append(abs(wasted_Rep - wasted_Dem))
        return output


def EG_2(df):
	#Generates efficiency gap from scratch, I think that we used to have some other processing for statistics that got lost in this version...

	final_partisan_vec = np.zeros(len(np.unique(df.CD_2010)))
	geo_partisan_vec = np.zeros(len(np.unique(df.CD_2010)))
	original_partisan_vec = np.zeros(len(np.unique(df.CD_2010)))
	
	for district in np.unique(df.district_final):
		final_partisan_vec[district] = (df.district_final == district).dot(df.PRES12_REP)/((df.district_final == district).dot(df.PRES12_REP + df.PRES12_DEM))
		geo_partisan_vec[district] = (df.district_final_alpha_0 == district).dot(df.PRES12_REP)/((df.district_final_alpha_0 == district).dot(df.PRES12_REP + df.PRES12_DEM))
		original_partisan_vec[district] = (df.CD_2010 == district).dot(df.PRES12_REP)/((df.CD_2010 == district).dot(df.PRES12_REP + df.PRES12_DEM))
	
	
	return {'geo': EG_helper(geo_partisan_vec),'original': EG_helper(original_partisan_vec), 'final': EG_helper(final_partisan_vec)}

def EG_helper(vec):
	wasted_Rep = 0
	wasted_Dem = 0
	for x in vec:
		if x < .5:
			wasted_Rep += x/len(vec)
			wasted_Dem += (.5-x)/len(vec)
		else:
			wasted_Rep += (x-.5)/len(vec)
			wasted_Dem += (1-x)/len(vec)
	return str(np.abs(wasted_Rep-wasted_Dem))

def make_histograms(df,state):
	make_folder('../maps/'+state+'/static')


	final_partisan_vec = np.zeros(len(np.unique(df.CD_2010)))
	geo_partisan_vec = np.zeros(len(np.unique(df.CD_2010)))
	original_partisan_vec = np.zeros(len(np.unique(df.CD_2010)))
	
	for district in np.unique(df.district_final):
		final_partisan_vec[district] = (df.district_final == district).dot(df.REP)/((df.district_final == district).dot(df.REP + df.DEM))
		geo_partisan_vec[district] = (df.district_final_alpha_0 == district).dot(df.REP)/((df.district_final_alpha_0 == district).dot(df.REP + df.DEM))
		original_partisan_vec[district] = (df.CD_2010 == district).dot(df.REP)/((df.CD_2010 == district).dot(df.REP + df.DEM))
	
	colors = sns.color_palette(n_colors=3, palette='bright')
	fig, ax = plt.subplots(1, 1,figsize=(10,7), subplot_kw=dict(aspect='auto'))
	ax.yaxis.set_visible(False)
	sns.distplot(geo_partisan_vec, color=colors[0], hist=False, label='Geographic Distance', rug=False, kde_kws={'bw':.2,  'gridsize':150, 'clip':(.25, .75), 'shade':False}, ax=ax)
	sns.distplot(final_partisan_vec, color=colors[1], hist=False, label='Geographic and Demographic Distance', rug=False, kde_kws={'bw':.2,  'gridsize':150, 'clip':(.25, .75), 'shade':False},ax=ax)
	sns.distplot(original_partisan_vec, color=colors[2], hist=False, label='Original Districts', rug=False, kde_kws={'bw':.2,  'gridsize':150, 'clip':(.25, .75), 'shade':False}, ax=ax)
	ax.legend(fontsize=16)
	ax.tick_params(axis='both', which='major', labelsize=15)
	ax.tick_params(axis='both', which='minor', labelsize=15)


	fig.savefig('../maps/'+state+'/static/partisan_outcomes_demographic.pdf', bbox_inches='tight', dpi=300)


	final_demog_vec = np.zeros(len(np.unique(df.CD_2010)))
	geo_demog_vec = np.zeros(len(np.unique(df.CD_2010)))
	original_demog_vec = np.zeros(len(np.unique(df.CD_2010)))
	
	for district in np.unique(df.district_final):
		final_demog_vec[district] = (df.district_final == district).dot(df.POP_BLACK)/((df.district_final == district).dot(df.POP_TOTAL))
		geo_demog_vec[district] = (df.district_final_alpha_0 == district).dot(df.POP_BLACK)/((df.district_final_alpha_0 == district).dot(df.POP_TOTAL))
		original_demog_vec[district] = (df.CD_2010 == district).dot(df.POP_BLACK)/((df.CD_2010 == district).dot(df.POP_TOTAL))
	
	
	fig, ax = plt.subplots(1, 1,figsize=(10,7), subplot_kw=dict(aspect='auto'))
	ax.yaxis.set_visible(False)
	sns.distplot(geo_demog_vec,color=colors[0],hist=False,label='Geographic Distance',rug=True,kde_kws={'bw':.2, 'gridsize':150,'clip':(0,.8),'shade':False},ax=ax)
	sns.distplot(final_demog_vec,color=colors[1],hist=False,label='Geographic and Demographic Distance',rug=True,kde_kws={'bw':.2, 'gridsize':150,'clip':(0,.8),'shade':False},ax=ax)
	sns.distplot(original_demog_vec,color=colors[2],hist=False,label='Original Districts',rug=True,kde_kws={'bw':.2, 'gridsize':150,'clip':(0,.8),'shade':False},ax=ax)
	ax.legend(fontsize=16)
	ax.tick_params(axis='both', which='major', labelsize=15)
	ax.tick_params(axis='both', which='minor', labelsize=15)	
	fig.savefig('../maps/'+state+'/static/demographic_outcomes_demographic.pdf', bbox_inches='tight', dpi=300)

	print('dumping')
	with open('../maps/'+state+'/static/kde_data_'+state+'.p', 'wb') as f:
		pickle.dump([geo_partisan_vec, final_partisan_vec, original_partisan_vec, geo_demog_vec, final_demog_vec, original_demog_vec], f)



def print_stats(df,state,alpha):
	#prints stats about the states. This includes values for population differences.
	final_pop_vec = np.zeros(len(np.unique(df.CD_2010)))
	geo_pop_vec = np.zeros(len(np.unique(df.CD_2010)))
	original_pop_vec = np.zeros(len(np.unique(df.CD_2010)))


	for district in np.unique(df.district_final):
		final_pop_vec[district] = (df.district_final == district).dot(df.POP_TOTAL)
		geo_pop_vec[district] = (df.district_final_alpha_0 == district).dot(df.POP_TOTAL)
		original_pop_vec[district] = (df.CD_2010==district).dot(df.POP_TOTAL)

	final_pop_pct_diff = (np.max(final_pop_vec)-np.min(final_pop_vec))/np.sum(final_pop_vec)

	
	geo_pop_pct_diff = (np.max(geo_pop_vec)-np.min(geo_pop_vec))/np.sum(geo_pop_vec)


	original_pop_pct_diff = (np.max(original_pop_vec)-np.min(original_pop_vec))/np.sum(original_pop_vec)


	GapDict = EG_2(df)




	with open('../maps/'+state+'/static/parameter_info.txt','w') as f:
		f.write('Original population deviation was ' + str(original_pop_pct_diff) + '\n')
		f.write('Original EG was ' + GapDict['original'] + '\n')
		f.write('Geographic population deviation was ' + str(geo_pop_pct_diff)+ '\n')
		f.write('Geographic EG was ' + GapDict['geo'] + '\n')
		f.write('Geographic and demographic population deviation was ' + str(final_pop_pct_diff) + '\n')
		f.write('Final EG was ' + GapDict['final'] + '\n')
		f.write('Final alphaW value was ' + str(alpha)+ '\n')
	


if __name__ == '__main__':

	cost_df = pd.DataFrame()
	state_list = list(states.keys())
	state_list.sort(key=itemgetter(0))

	state_results = {}
	hist_labels = ['Current', r"$\alpha_W=0$", r"$\alpha_W=.25$", r"$\alpha_W=.75$"]

	state_df_list = []
	district_results = pd.DataFrame()	
	# for state in state_list:
	for state in state_list:
		print(state)	

		# get data from shapefiles if not available already
		ds = district_solver(state)
		ds.load_data(build=True)
		ds.get_optimal_districts()
		
		state_results[state] = {'cost0': ds.cost0_best,
		 						'cost_best': ds.cost_best,
		 						'n_districts': ds.n_districts,
		 						'n_precincts': ds.n_precincts,
		 						'population': ds.pcnct_df.POP_TOTAL.sum(),
		 						'rep_votes': (ds.pcnct_df[['PRES12_REP', 'PRES08_REP']].values.sum())/2,
		 						'dem_votes': (ds.pcnct_df[['PRES12_DEM', 'PRES08_DEM']].values.sum())/2
		 						}

		ds.pcnct_df['state'] = state


		# plot functions group precinct dataframe into district dataframe
		if state in ['NC', 'MD', 'VA']:
			is_wide = True
		else:
			is_wide = False
		
		# make maps			
		mapper = map_maker(ds.pcnct_df, state, is_wide=is_wide)
		mapper.make_state_maps('../maps/{}/static/'.format(state))

		# compute district level stats
		dist_df_before = ds.pcnct_df[['DEM', 'REP', 'POP_BLACK', 'POP_TOTAL', 'CD_2010']].groupby('CD_2010').sum()
		dist_df_before.loc[:, 'precinct_cost'] = ds.pcnct_df['precinct_cost_0'].values[0]
		dist_df_before.loc[:, 'current_districts'] = True

		dist_df_final = ds.pcnct_df[['DEM', 'REP', 'POP_BLACK', 'POP_TOTAL', 'district_final']].groupby('district_final').sum()
		dist_df_final.loc[:, 'precinct_cost'] = ds.pcnct_df['precinct_cost_final'].values[0]
		dist_df_final.loc[:, 'current_districts'] = False
		
		# combine aggregate info from before
		df = dist_df_before.append(dist_df_final)
		df['DEM_PCT'] = df['DEM']/df[['DEM', 'REP']].sum(axis=1)
		df['REP_PCT'] = df['DEM']/df[['DEM', 'REP']].sum(axis=1)
		df['state'] = state
		district_results = district_results.append(df)

		# make some histograms and look at other stats
		make_histograms(ds.pcnct_df, state)
		
		print_stats(ds.pcnct_df, state, ds.alphaW_best)
		
		#This will pickle the file, for testing purposes 3/26/18
		#with open('tmp','wb') as f:
			#pickle.dump(ds,f)

                # TODO here: make histograms for demographics and partisan outcomes
                # The code that will probably do this in a pretty way can be found at https://seaborn.pydata.org/tutorial/distributions.html
                #TODO: compute the population variance between districts, so that it can be reported in figures.		


	district_results.to_csv('precinct_level_aggregates.csv')
