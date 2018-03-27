import pandas as pd
import numpy as np

import pickle

from make_districts import district_solver
from states import states
from make_maps import map_maker

import seaborn as sns
import matplotlib.pyplot as plt


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
	fig.savefig(filename, bbox_inches='tight', dpi=100)
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
        #Generates and saves static histograms from the dataframe

        #Histogram for partisan outcomes

	final_partisan_vec = np.zeros(len(np.unique(df.CD_2010)))
	geo_partisan_vec = np.zeros(len(np.unique(df.CD_2010)))
	original_partisan_vec = np.zeros(len(np.unique(df.CD_2010)))
	
	for district in np.unique(df.district_final):
		final_partisan_vec[district] = (df.district_final == district).dot(df.PRES12_REP)/((df.district_final == district).dot(df.PRES12_REP + df.PRES12_DEM))
		geo_partisan_vec[district] = (df.district_final_alpha_0 == district).dot(df.PRES12_REP)/((df.district_final_alpha_0 == district).dot(df.PRES12_REP + df.PRES12_DEM))
		original_partisan_vec[district] = (df.CD_2010 == district).dot(df.PRES12_REP)/((df.CD_2010 == district).dot(df.PRES12_REP + df.PRES12_DEM))
	
	
	fig, ax = plt.subplots(1, 1,figsize=(8,8), subplot_kw=dict(aspect='auto'))
	ax.yaxis.set_visible(False)
	sns.distplot(geo_partisan_vec,color='b',hist=False,label='Geographic distance',rug=True,kde_kws={'bw':.2, 'gridsize':150,'clip':(.25,.75),'shade':False},ax=ax)
	sns.distplot(final_partisan_vec,color='r',hist=False,label='Geographic and demographic distance',rug=True,kde_kws={'bw':.2, 'gridsize':150,'clip':(.25,.75),'shade':False},ax=ax)
	sns.distplot(original_partisan_vec,color='g',hist=False,label='Original districts',rug=True,kde_kws={'bw':.2, 'gridsize':150,'clip':(.25,.75),'shade':False},ax=ax)

	fig.savefig('../maps/'+state+'/static/partisan_outcomes_demographic.png',bbox_inches='tight')


	final_demog_vec = np.zeros(len(np.unique(df.CD_2010)))
	geo_demog_vec = np.zeros(len(np.unique(df.CD_2010)))
	original_demog_vec = np.zeros(len(np.unique(df.CD_2010)))
	
	for district in np.unique(df.district_final):
		final_demog_vec[district] = (df.district_final == district).dot(df.POP_BLACK)/((df.district_final == district).dot(df.POP_TOTAL))
		geo_demog_vec[district] = (df.district_final_alpha_0 == district).dot(df.POP_BLACK)/((df.district_final_alpha_0 == district).dot(df.POP_TOTAL))
		original_demog_vec[district] = (df.CD_2010 == district).dot(df.POP_BLACK)/((df.CD_2010 == district).dot(df.POP_TOTAL))
	
	
	fig, ax = plt.subplots(1, 1,figsize=(8,8), subplot_kw=dict(aspect='auto'))
	ax.yaxis.set_visible(False)
	sns.distplot(geo_demog_vec,color='b',hist=False,label='Geographic distance',rug=True,kde_kws={'bw':.2, 'gridsize':150,'clip':(0,.8),'shade':False},ax=ax)
	sns.distplot(final_demog_vec,color='r',hist=False,label='Geographic and demographic distance',rug=True,kde_kws={'bw':.2, 'gridsize':150,'clip':(0,.8),'shade':False},ax=ax)
	sns.distplot(original_demog_vec,color='g',hist=False,label='Original districts',rug=True,kde_kws={'bw':.2, 'gridsize':150,'clip':(0,.8),'shade':False},ax=ax)

	fig.savefig('../maps/'+state+'/static/demographic_outcomes_demographic.png',bbox_inches='tight')




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
	state_list.sort()

	state_results = {}
	hist_labels = ['Current', r"$\alpha_W=0$", r"$\alpha_W=.25$", r"$\alpha_W=.75$"]

	state_df_list = []
	district_results = pd.DataFrame()	
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
		mapper = map_maker(ds.pcnct_df, state)
		mapper.make_state_maps()


		make_histograms(ds.pcnct_df, state)
		
		print_stats(ds.pcnct_df,state,ds.alphaW_best)
		
		#This will pickle the file, for testing purposes 3/26/18
		#with open('tmp','wb') as f:
			#pickle.dump(ds,f)


                # TODO here: make histograms for demographics and partisan outcomes
                # The code that will probably do this in a pretty way can be found at https://seaborn.pydata.org/tutorial/distributions.html

                #TODO: compute the population variance between districts, so that it can be reported in figures.

		dist_df0 = mapper._get_district_df('CD_2010')
		dist_df0.rename(columns={'precinct_cost_0':'precinct_cost'}, inplace=True)
		dist_df0['current_districts'] = True

		dist_df_final = mapper._get_district_df('district_final')
		dist_df_final.rename(columns={'precinct_cost_final':'precinct_cost'}, inplace=True)
		dist_df_final['current_districts'] = False

		# which columns to keep
		columns = ['precinct_cost', 'current_districts', 'DEM', 'REP', 'DEM_PCT', 'REP_PCT']
		# columns += [c for c in dist_df0.columns if 'PRES' in c]
		columns += [c for c in dist_df0.columns if 'POP_' in c]
		
		df = dist_df0.append(dist_df_final)
		df = df[columns]
		df['state'] = state
		district_results = district_results.append(df)

	district_results.to_csv('map_test_data.csv')
