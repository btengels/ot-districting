import os
import numpy as np
import pandas as pd
import geopandas as geo
import pickle
import utils 

from subprocess import call
from glob import glob
import transport_plan_functions as tpf
from states import states


class district_solver(object):
	"""Object that contains all the data/methods needed to create optimal districts"""

	def __init__(self, state):
		self.state = state
		self.state_name = states[self.state][0]
		self.wget = False

		self.random_start = True
		self.reg_param = 20 #25

	def load_data(self, build=False):
		"""Assigns self.pcnct_df attribute, either by reading in a pickle or solving from raw data

		"""
		if build:
			self._get_state_data()

		else:
			try:
				self.pcnct_df = pd.read_pickle('../Data-Files/' + self.state + '/precinct_data.p')
			except:
				self._get_self.state_data()		
					

	def get_optimal_districts(self, tmp_reg=.5):
		"""
		This function takes a geopandas DataFrame and computes the set of 
		congressional districts that minimize the total "distance" between precincts
		(rows in DataFrame) and the district "offices" (centroids).

		INPUT: 
		----------------------------------------------------------------------------
		reg: scalar, regularization term used in cost function

		OUTPUT:
		----------------------------------------------------------------------------
		cost0: scalar, value of cost function at F_loc0
		cost: scalar, value of cost function at algorithm end
		pcnct_df: geopandas dataFrame, now with additional column 'final_district'
		F_opt: np.array, indicates the location of optimal office locations
		"""
		# weight each precinct by its population share
		n_districts = int(self.pcnct_df.CD_2010.max()) + 1
		n_precincts = len(self.pcnct_df)
		pcnct_loc = self.pcnct_df[['INTPTLON10', 'INTPTLAT10', 'BLACK_PCT']].values
		pcnct_pop = np.maximum(self.pcnct_df.POP_TOTAL.values, 20) # put nonzero mass in every precinct
		pcnct_wgt = pcnct_pop/pcnct_pop.sum()

		# initial guess for district offices
		office_loc_list = []

		if self.random_start:
			num_starts = 25

			# randomly select initial districts, all districts have equal weight
			for i in range(num_starts):
				office = pcnct_loc[np.random.randint(0, n_precincts, n_districts)]

				office_loc_list.append(office) 
			
		else:
			# use most populated precinct of current district as initial guess
			office_loc0 = np.zeros((n_districts, 3))
			for i in range(n_districts):

				df = self.pcnct_df[self.pcnct_df['CD_2010'].values==i]

				# df['pop_area'] = df.POP_TOTAL/(df.area*1000)
				pop_argmax = df['POP_TOTAL'].argmax()

				office_loc0[i, 0] = df['INTPTLON10'].loc[pop_argmax]
				office_loc0[i, 1] = df['INTPTLAT10'].loc[pop_argmax]
				office_loc0[i, 2] = df['POP_BLACK'].sum()/df['POP_TOTAL'].sum()	
			
			office_loc_list.append(office_loc0) 

		# office weights (each office has the same share of population) 	
		F_wgt = np.ones((n_districts,))/n_districts

		# initial transport plan: i.e. the measure of pop assigned to each district 
		transp_map = np.zeros((n_precincts, n_districts)) 
		for i_d, dist in enumerate(self.pcnct_df.CD_2010.values):
			transp_map[i_d, int(dist)] = pcnct_wgt[i_d]	




		# Begin by solving for best districts when alphaW = 0.
		alphaW = 0
		contiguity = True
		self.cost_best = 20


		for F_loc0 in office_loc_list:
				
			# distance of each precinct to its center (aka, "office")
			DistMat = tpf.get_DistMat(pcnct_loc, F_loc0, alphaW)
			Dist_travel, Dist_demographics = tpf.distance_metric(pcnct_loc, F_loc0, alphaW)

			# evaluate cost function given current districts
			office_loc = tpf.optimizeF(pcnct_loc, pcnct_wgt, F_loc0, F_wgt, transp_map, DistMat)

			temp = np.log(transp_map)	
			temp[np.isreal(np.log(transp_map))] = 0
			cost0 = np.sum(DistMat*transp_map) + 1.0/self.reg_param*np.sum(temp*transp_map)

			# compute optimal districts, offices, and transport cost
			opt_dist, F_opt, cost, transp_map_opt = tpf.gradientDescentOT(pcnct_loc, 
													    			  pcnct_wgt, 
														  			  office_loc, 
														  			  F_wgt,
														  			  reg=self.reg_param,
														  			  alphaW=alphaW)

			temp = np.log(transp_map_opt)	
			temp[np.isreal(np.log(transp_map_opt))] = 0

			Dist_travel, Dist_demographics = tpf.distance_metric(pcnct_loc, F_opt, alphaW)
			cost_travel = np.sum(Dist_travel*transp_map_opt)#Note I changed the alphaW here...
			cost_demographic = alphaW * np.sum(Dist_demographics*transp_map_opt) 
			cost_regularization = 1.0/self.reg_param*np.sum(temp*transp_map_opt)

			# update if we are the current best district and contiguous
			if cost < self.cost_best:
				self.opt_district_best = opt_dist.copy()
				self.F_opt_best = F_opt.copy()
				self.F_loc0_best = F_loc0
				self.cost0_best = cost0
				self.cost_best = cost
				self.alphaW_best = alphaW

		print(self.state, alphaW, self.cost_best)


		# update dataframe with districts for each precinct
		self.pcnct_df['district_final_alpha_0'] = self.opt_district_best
		
		DistMat_0 = tpf.distance_metric(pcnct_loc, self.F_loc0_best, 0)		
		self.pcnct_df['precinct_cost_alpha_0_init'] = self.cost0_best
		self.pcnct_df['precinct_cost_alpha_0_final'] = self.cost_best

		self.n_districts = n_districts
		self.n_precincts = n_precincts
	





		#Next increase alphaW until the 10% threshold has been reached. 10% may need to be made higher?

		for alphaW in [.4, .6]:#I trimmed some values for speed here.
			
			# initialize cost_best variable

			self.cost_best = 20

			# find the best result over (perhaps several random) starting points
			for F_loc0 in office_loc_list:
				
				# distance of each precinct to its center (aka, "office")
				DistMat = tpf.get_DistMat(pcnct_loc, F_loc0, alphaW)
				Dist_travel, Dist_demographics = tpf.distance_metric(pcnct_loc, F_loc0, alphaW)

				# evaluate cost function given current districts
				office_loc = tpf.optimizeF(pcnct_loc, pcnct_wgt, F_loc0, F_wgt, transp_map, DistMat)

				temp = np.log(transp_map)	
				temp[np.isreal(np.log(transp_map))] = 0
				cost0 = np.sum(DistMat*transp_map) + 1.0/self.reg_param*np.sum(temp*transp_map)

				# compute optimal districts, offices, and transport cost
				opt_dist, F_opt, cost, transp_map_opt = tpf.gradientDescentOT(pcnct_loc, 
														    			  	  pcnct_wgt, 
															  			  	  office_loc, 
															  			  	  F_wgt,
															  			  	  reg=self.reg_param,
															  			  	  alphaW=alphaW)

				temp = np.log(transp_map_opt)	
				temp[np.isreal(np.log(transp_map_opt))] = 0

				Dist_travel, Dist_demographics = tpf.distance_metric(pcnct_loc, F_opt, alphaW)
				cost_travel = np.sum(Dist_travel*transp_map_opt)#Note I changed the alphaW here...
				cost_demographic = alphaW * np.sum(Dist_demographics*transp_map_opt) 
				cost_regularization = 1.0/self.reg_param*np.sum(temp*transp_map_opt)

				# update if we are the current best district and contiguous
				if cost < self.cost_best:
					self.opt_district_best = opt_dist.copy()
					self.F_opt_best = F_opt.copy()
					self.F_loc0_best = F_loc0
					self.cost0_best = cost0
					self.cost_best = cost
					self.alphaW_best = alphaW
					self.best_cost_demographic = cost_demographic

			print(self.state, alphaW, self.cost_best)


			if self.best_cost_demographic/self.cost_best > .1:
				print(self.state, alphaW, self.cost_best, self.best_cost_demographic/self.cost_best)
				break  # exit alphaW loop

		# update dataframe with districts for each precinct
		self.pcnct_df['district_final'] = self.opt_district_best
		
		DistMat_0 = tpf.distance_metric(pcnct_loc, self.F_loc0_best, 0)		
		self.pcnct_df['precinct_cost_0'] = self.cost0_best
		self.pcnct_df['precinct_cost_final'] = self.cost_best

		self.n_districts = n_districts
		self.n_precincts = n_precincts

			
	def _get_state_data(self):
		"""
		This function downloads the data from autoredistrict.org's ftp. After some 
		minor cleaning, the data is saved as a geopandas DataFrame.

		NOTE: currently the shape files on autoredistrict's ftp are districts
		instead of precincts as before. Don't use wget. 

		INPUT:
		----------------------------------------------------------------------------
		self.state: string, postal ID for self.state and key to "self.states" dictionary
		wget: boolian (default=False), whether to download new shape files.

		OUTPUT:
		----------------------------------------------------------------------------
		None, but DataFrame is pickled in ../Data-Files/<self.state> alongside the shape
		files.
		"""
		# make folder if it doesn't already exist
		prefix = '../Data-Files/' + self.state
		utils.make_folder(prefix)

		# import shape files
		url = 'ftp://autoredistrict.org/pub/shapefiles2/' + self.state_name + '/2010/2012/vtd/tl*'
		if self.wget:
		    call(['wget', '-P', prefix,
		          ])

		# read shape files into geopandas
		geo_path = glob(prefix + '/tl*.shp')[0]
		geo_df = geo.GeoDataFrame.from_file(geo_path)
		geo_df.CD_2010 = geo_df.CD_2010.astype(int)

		# drops totals and other non-precinct observations
		geo_df = geo_df[geo_df.CD_2010 >= 0]

		# -------------------------------------------------------------------------
		# ADJUST ASPECT RATIO HERE:
		# -------------------------------------------------------------------------
		geo_df = geo_df.to_crs(epsg=4267)
		# geo_df.geometry = geo_df.geometry.scale(xfact=1/100000, yfact=1/100000, zfact=1.0, origin=(0, 0))

		# simplify geometries for faster image rendering
		# bigger number gives a smaller file size    
		geo_df.geometry = geo_df.geometry.simplify(.01).buffer(.007)

		# add longitude and latitude
		lonlat = np.array([t.centroid.coords.xy for t in geo_df.geometry])
		geo_df['INTPTLON10'] = lonlat[:, 0]
		geo_df['INTPTLAT10'] = lonlat[:, 1]

		# -------------------------------------------------------------------------
		# make sure congressional districts are numbered starting at 0
		geo_df.CD_2010 -= geo_df.CD_2010.min()

		# correct a few curiosities
		if self.state in ['KY']:
		    geo_df.drop(['POP_BLACK',
		    			 'POP_WHITE',
		    			 'POP_ASIAN',
		    			 'POP_HAWAII',
		                 'POP_HISPAN',
		                 'POP_INDIAN',
		                 'POP_MULTI',
		                 'POP_OTHER',
		                 'POP_TOTAL'], axis=1, inplace=True)

		    geo_df.rename(index=str, columns={'VAP_BLACK': 'POP_BLACK',
		                                      'VAP_WHITE': 'POP_WHITE',
		                                      'VAP_ASIAN': 'POP_ASIAN',
		                                      'VAP_HAWAII': 'POP_HAWAII',
		                                      'VAP_HISPAN': 'POP_HISPAN',
		                                      'VAP_INDIAN': 'POP_INDIAN',
		                                      'VAP_MULTI': 'POP_MULTI',
		                                      'VAP_OTHER': 'POP_OTHER',
		                                      'VAP_TOT': 'POP_TOTAL'},
		                  inplace=True)

		# percent black in each precinct, account for precincts with zero population
		geo_df['BLACK_PCT'] = np.maximum(geo_df['POP_BLACK']/geo_df['POP_TOTAL'], 0)
		geo_df.loc[np.isfinite(geo_df['POP_TOTAL']) == False, 'BLACK_PCT'] = 0
		geo_df.fillna({'BLACK_PCT':0}, inplace=True)

		# political breakdown
		geo_df['DEM'] = geo_df[['PRES04_DEM','PRES08_DEM','PRES12_DEM']].mean(axis=1)
		geo_df['REP'] = geo_df[['PRES04_REP','PRES08_REP','PRES12_REP']].mean(axis=1)
		geo_df['REP_PCT'] = geo_df['REP']/(geo_df['DEM'] + geo_df['REP'])
		geo_df['DEM_PCT'] = geo_df['DEM']/(geo_df['DEM'] + geo_df['REP'])

		# unpack multipolygons
		self.pcnct_df = geo_df#utils.unpack_multipolygons(geo_df)

		# pickle dataframe for future use
		pickle.dump(self.pcnct_df, open(prefix + '/precinct_data.p', 'wb'), protocol=2)


