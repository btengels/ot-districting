import utils
import numpy as np
import shapely

import matplotlib as mpl
mpl.use('TkAgg')   # if using mac OS X
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from bokeh.io import output_file
from bokeh.models import ColumnDataSource as CDS
from bokeh.plotting import *
from bokeh.models import HoverTool


class map_maker(object):

	def __init__(self, pcnct_df, state):
		"""
		INPUTS:
		----------------------------------------------------------------------------
		pcnct_df: geopandas DataFrame

		"""
		self.pcnct_df = pcnct_df
		self.palette = self._make_palette()
		np.random.shuffle(self._make_palette())
		self.state = state

	def make_state_maps(self):
		"""
		This function takes a state postal code and fetches the corresponding 
		pickled dataframe with precinct level data and geometries. The funciton then
		solves for the optimal district, and makes both static and bokeh-style plots
		for the state based on its current and optimal congressional districts.

		INPUTS: 
		----------------------------------------------------------------------------
		state: string, key to "states" dictionary
		random_start: boolean (default=False), whether to use random coordinates for
					  office locations (and choose the best of several iterations)

		OUTPUTS: 
		----------------------------------------------------------------------------
		df_list: python list, contains geopandas DataFrames. One for each value of 
				 alphaW. This will eventually just hold 2 dataframes for the current 
				 and optimal set of districts
		"""
		# make map folders if not existent
		utils.make_folder('../maps/' + self.state)
		utils.make_folder('../maps/' + self.state + '/static')
		utils.make_folder('../maps/' + self.state + '/dynamic')	

		# make figure/axis objects and plot	initial figure
		fig, ax = plt.subplots(1, 1, subplot_kw=dict(aspect='equal'))

		# make sure all plots have same bounding boxes
		xlim = (self.pcnct_df.geometry.bounds.minx.min(), self.pcnct_df.geometry.bounds.maxx.max())
		ylim = (self.pcnct_df.geometry.bounds.miny.min(), self.pcnct_df.geometry.bounds.maxy.max())				
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)

		# list to save district-level DataFrames
		df_list = []

		# open/save figure for current districts
		filename = '../maps/' + self.state + '/static/before.png'
		current_dists = self.pcnct_df['CD_2010'].values.astype(int)
		colors = np.array([self.palette[i] for i in current_dists])
		patches = self.plot_state(colors, ax, fig, filename)
		
		# save html figure for current districts
		filename = '../maps/' + self.state + '/dynamic/before.html'
		district_df = self.make_bokeh_map('CD_2010', filename)
		df_list.append(district_df)

		# check to see if contiguity is broken in initial districts (islands?, etc.)
		# contiguity = utils.check_contiguity(self.pcnct_df, 'CD_2010')

		# update colors on existing figure for optimal districting solution
		colors = np.array([self.palette[i] for i in self.pcnct_df['district_final']])
		patches.set_color(colors)

		# make sure bounding box is consistent across figures
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)

		# # plot district offices and save figure
		# stars =	ax.scatter(self.F_opt[:, 0], self.F_opt[:, 1],
		# 				   color='black',
		# 				   marker='*', 
		# 				   s=30, 
		# 				   alpha=.7)	

		prefix = '../maps/' + self.state + '/static/'
		filename = prefix + '_after.png'
		fig.savefig(filename, bbox_inches='tight', dpi=300)
		# stars.remove()

		# make bokeh map
		prefix = '../maps/' + self.state + '/dynamic/'
		filename =  prefix + '_after.html'
		df = self.make_bokeh_map('district_final', filename)

	def plot_state(self, colors, ax, fig, filename, F_opt=None):
		"""
		Function takes geopandas DataFrame and plots the precincts colored according
		to their congressional district. Saves figure at path "filename."

		INPUTS:
		----------------------------------------------------------------------------
		district_group: string, column name in pcnct_df which contains each precinct's 
						congressional district
		ax: matplotlib axis object
		fig: matplotlib figure object, used to save final figure
		filename: string, path to saved figure
		F_opt: np.array, plots the location of district offices if provided
		"""	

		# plot patches, colored according to district
		patches = self.plot_patches(self.pcnct_df.geometry.values, colors, ax, lw=.1)
		
		# plot stars for office locations
		if F_opt is not None:
			for i in range(len(F_opt)):
				ax.scatter(F_opt[i, 0], F_opt[i, 1], color='black', marker='*', s=30, alpha=1)	

		if filename is not None:
			ax.set_yticklabels([])
			ax.set_xticklabels([])
			fig.savefig(filename, bbox_inches='tight', dpi=100)
			plt.close()			

		return patches


	def make_bokeh_map(self, groupvar, filename):
		"""
		This file makes a map using the Bokeh library and a GeoPandas DataFrame.

		INPUTS:
		----------------------------------------------------------------------------
		groupvar: string, column name identifying congressional districts
		filename: string, destination of bokeh plot

		OUTPUTS:	
		----------------------------------------------------------------------------
		df: pandas DataFrame, contrains district-level information
		"""		
		output_file(filename)		

		df = self._get_district_df(groupvar)

		source = CDS(data=dict(x = df['patchx'].values.astype(list),
							   y = df['patchy'].values.astype(list),
							   color1 = df['color1'].values.astype(list),
							   dist = df['dist'].values.astype(list),
							   pop_pct  = df['POP_PCT'].values.astype(list),
							   black_pct  = df['BLACK_PCT'].values.astype(list),
							   hisp_pct  = df['HISPAN_PCT'].values.astype(list),		
							   rep_pct  = df['REP_PCT'].values.astype(list),
							   dem_pct  = df['DEM_PCT'].values.astype(list),
							   )
				    )

		# adjust image size according to shape of state
		lat_range = self.pcnct_df.INTPTLAT10.max() - self.pcnct_df.INTPTLAT10.min()
		lon_range = self.pcnct_df.INTPTLON10.max() - self.pcnct_df.INTPTLON10.min()
		
		# make bokeh figure. 
		TOOLS = "pan,wheel_zoom,box_zoom,reset,hover"	
		p = figure(plot_width=420,
				   plot_height=int(420*(lat_range/lon_range)*1.1),
				   tools=TOOLS,
				   toolbar_location='above')

		# hover settings
		hover = p.select_one(HoverTool)
		hover.point_policy = "follow_mouse"
		hover.tooltips = [("District", "@dist"),
						  ("Population Share", "@pop_pct%"),
						  ("Black Pop.", "@black_pct%"),
						  ("Hispanic Pop.", "@hisp_pct%"),
				          ("Democrat", "@dem_pct%"),
				          ("Republican", "@rep_pct%")]
		
		# remove bokeh logo
		p.toolbar.logo = None
		p.patches('x', 'y', source=source, 
		          fill_color='color1', fill_alpha=.9, 
		          line_color='black', line_width=.5,
		          line_alpha=.4)

		# Turn off tick labels
		p.axis.major_label_text_font_size = '0pt'  
		
		# Turn off tick marks 	
		p.grid.grid_line_color = None
		p.outline_line_color = "black"
		p.outline_line_width = .5
		p.outline_line_alpha = 1
		# p.background_fill_color = "gray"
		p.background_fill_alpha = .1
		p.axis.major_tick_line_color = None  # turn off major ticks
		p.axis[0].ticker.num_minor_ticks = 0  # turn off minor ticks
		p.axis[1].ticker.num_minor_ticks = 0

		# save output as html file	
		# show(p)
		save(p)

		# return district level DataFrame
		return df.groupby('dist').first()


	def plot_patches(self, geoms, colors, ax, lw=.1):
		"""
		Plots precincts colored according to their congressional district.
		Uses matplotlib's PathCollection rather than geopandas' native plot() 
		function.

		INPUTS: 
		----------------------------------------------------------------------------
		geos: np.array, vector of polygons
		colors: np.array, vector indicating the color of each polygon	
		ax: matplotlib axis object
		lw: scalar, line width

		OUTPUTS: 
		----------------------------------------------------------------------------
		patches: matplotlib PathCollection object, we return this so we can plot the 
				 map once and then only worry about updating the colors later. 
		"""

		# make list of polygons (make sure they only have x and y coordinates)
		patches = []
		for poly in geoms:		
			a = np.asarray(poly.exterior)
			if poly.has_z:
				poly = shapely.geometry.Polygon(zip(*poly.exterior.xy))
			patches.append(Polygon(a))

		# make PatchCollection object
		patches = PatchCollection(patches)

		# set colors and linewidth
		patches.set_lw(lw)
		patches.set_color(colors)

		# plot on ax
		ax.add_collection(patches, autolim=True)
		ax.autoscale_view()

		return patches


	def _make_palette(self, cmap=plt.cm.Paired, hex=False):
		"""
		Takes matplotlib cmap object and generates a palette of n equidistant points
		over the cmap spectrum, returned as a list. 
		
		INPUTS:
		----------------------------------------------------------------------------
		n_districts: int, number of districts
		cmap: matplotlib colormap object, e.g. cmap = plt.cm.Paired
		hex: boolean (default=False), If true returns colors as hex strings instead 
			 of rgba tuples.

		OUTPUTS:
		----------------------------------------------------------------------------
		palette: list, list of size n_districts of colors
		"""
		n_districts = len(self.pcnct_df.CD_2010.unique())

		# define the colormap
		cmaplist = [cmap(i) for i in range(cmap.N)]		
		palette =[cmaplist[i] for i in range(0, cmap.N, int(cmap.N/n_districts))]	

		if hex is True:
			palette = [mpl.colors.rgb2hex(p) for p in palette]

		return palette


	def _get_district_df(self, groupvar):
		"""

		"""
		pcnct_pop = np.maximum(self.pcnct_df.POP_TOTAL.values, 20)
		# pcnct_wgt = pcnct_pop/pcnct_pop.sum()

		# self.pcnct_df['precinct_cost_0'] = self.pcnct_df['precinct_cost_0'] * pcnct_wgt		
		# self.pcnct_df['precinct_cost_final'] = self.pcnct_df['precinct_cost_final'] * pcnct_wgt		

		# aggregate precincts by district
		df = self.pcnct_df.dissolve(by=groupvar, aggfunc=np.sum)

		# aggregating can create multi-polygons, can't plot in bokeh so unpack those
		df = utils.unpack_multipolygons(df, impute_vals=False)
		df = df[df.geom_type == 'Polygon']
		df.geometry = [shapely.geometry.polygon.asPolygon(g.exterior) for g in df.geometry.values]

		# smooth out the district level polygons
		df.geometry = df.geometry.simplify(.015).buffer(.02)
		df = df[df.geom_type == 'Polygon']

		# remove bleed (nonempty intersection) resulting from buffer command
		for ig, g in enumerate(df.geometry):
			for ig2, g2 in enumerate(df.geometry):
				if ig != ig2:
					g -= g2
			df.geometry.iloc[ig] = g

		# aggregating can create multi-polygons, can't plot in bokeh so unpack those
		df = utils.unpack_multipolygons(df, impute_vals=False)
		df = df[df.geom_type == 'Polygon']
		df.geometry = [shapely.geometry.polygon.asPolygon(g.exterior) for g in df.geometry.values]		

		df['dist'] = df.index.values.astype(int) + 1
		df.drop_duplicates(subset='dist', inplace=True)		

		# carry over important variables into the district-level dataframe 
		df['area'] = df.geometry.area	
		df['DEM'] = df[['PRES04_DEM','PRES08_DEM','PRES12_DEM']].mean(axis=1)
		df['REP'] = df[['PRES04_REP','PRES08_REP','PRES12_REP']].mean(axis=1)

		# district level variables
		df['BLACK_PCT'] = (df['POP_BLACK'].values/df['POP_TOTAL'].values)*100
		df['HISPAN_PCT'] = (df['POP_HISPAN'].values/df['POP_TOTAL'].values)*100
		df['REP_PCT'] = (df['REP'].values/df[['REP','DEM']].sum(axis=1).values)*100
		df['DEM_PCT'] = (df['DEM'].values/df[['REP','DEM']].sum(axis=1).values)*100
		df['POP_PCT'] = (df['POP_TOTAL'].values/df['POP_TOTAL'].sum())*100
		
		df['n_precincts'] = len(self.pcnct_df)

		# variables for mapping
		df['patchx'] = df.geometry.apply(lambda x: utils.get_coords(x, xcoord=True))
		df['patchy'] = df.geometry.apply(lambda x: utils.get_coords(x, xcoord=False))
		df['color1'] = [self.palette[i-1] for i in df.dist]
		
		return df
