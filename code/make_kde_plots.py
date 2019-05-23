import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter

import pandas as pd
from states import states
import matplotlib as mpl
mpl.rc('font',family='serif')


def make_histograms(geo_partisan_vec, final_partisan_vec, original_partisan_vec, ax, clip):
	"""
	"""
	colors = sns.color_palette(n_colors=3, palette='bright')
	sns.distplot(geo_partisan_vec, ax=ax, color=colors[0], hist=False, label='Geographic Distance', rug=False, kde_kws={'bw':.2,  'gridsize':150, 'clip': clip, 'shade':False})
	sns.distplot(final_partisan_vec, ax=ax, color=colors[1], hist=False, label='Geographic and Demographic Distance', rug=False, kde_kws={'bw':.2,  'gridsize':150, 'clip': clip, 'shade':False})
	sns.distplot(original_partisan_vec, ax=ax, color=colors[2], hist=False, label='Original Districts', rug=False, kde_kws={'bw':.2,  'gridsize':150, 'clip': clip, 'shade':False})
	
	ax.tick_params(axis='both', which='major', labelsize=15)
	ax.tick_params(axis='both', which='minor', labelsize=15)
	ax.set_xlim(clip)
	ax.legend([], frameon=False)


if __name__ == "__main__":	

	state_list = list(states.keys())
	state_list.sort(key=itemgetter(0))
	state_list.remove('CO')

	fig, axes = plt.subplots(len(state_list), 1,figsize=(7, 17), sharex=False)

	for state, ax in zip(state_list, axes):

		with open('../maps/'+state+'_10pct/static/kde_data_'+state+'.p', 'rb') as f:
			[geo_partisan_vec, final_partisan_vec, original_partisan_vec, geo_demog_vec, final_demog_vec, original_demog_vec] = pickle.load(f)

		clip = (0, 1)
		make_histograms(geo_partisan_vec, final_partisan_vec, original_partisan_vec, ax, clip)
		ax.set_ylabel(state, fontsize=15)
		ax.set_yticks([])
		ax.set_yticklabels([])

	ax.set_xlabel('Percent Republican', fontsize=15)
	ax.legend(fontsize=14, ncol=1, bbox_to_anchor=(.95, -.36), frameon=False)
	fig.savefig('../maps/partisan_outcomes_allstates.pdf', bbox_inches='tight', dpi=500)

	for i in range(100):
		plt.close()


	fig, axes = plt.subplots(len(state_list), 1,figsize=(7,17), sharex=False)
	for state, ax in zip(state_list, axes):

		with open('../maps/'+state+'_10pct/static/kde_data_'+state+'.p', 'rb') as f:
			[geo_partisan_vec, final_partisan_vec, original_partisan_vec, geo_demog_vec, final_demog_vec, original_demog_vec] = pickle.load(f)

		clip = (-.03,1)
		make_histograms(geo_demog_vec, final_demog_vec, original_demog_vec, ax, clip)
		ax.set_ylabel(state, fontsize=15)
		ax.set_yticks([])
		ax.set_yticklabels([])

	# plt.show()
	ax.set_xlabel('Percent African American', fontsize=15)

	ax.legend(fontsize=14, ncol=1, bbox_to_anchor=(.95, -.36), frameon=False)
	fig.savefig('../maps/demographic_outcomes_allstates.pdf', bbox_inches='tight', dpi=500)
