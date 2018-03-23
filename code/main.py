import pandas as pd
import numpy as np

from make_districts import district_solver
from states import states
from make_maps import map_maker 


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
