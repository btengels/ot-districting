# new_maps.py

import pickle
import utils
import numpy as np
import geopandas as geo
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from states import states


class map_maker(object):

    def __init__(self, pcnct_df, state, is_wide=False):
        """
        INPUTS:
        ----------------------------------------------------------------------------
        pcnct_df : geopandas DataFrame
        state : state postal abbreviation
        state_shape : 'tall', 'square', or 'wide'

        """
        self.pcnct_df = pcnct_df.to_crs(epsg=4267)
        self.state = state

        if is_wide:
            self.figsize = (10, 3.2)
        else: 
            self.figsize = (10, 5)

    def make_state_maps(self, figure_directory=None):
        """Creates figure and calls '_get_single_map' and '_get_cmap' to fill panels

        INPUTS:
        ----------------------------------------------------------------------------        
        figure_directory : str - directory where map will be saved; if None (default), no figure is saved

        """        
        # initialize figure object: 2 subplots and a colorbar centered beneath
        fig = plt.figure(figsize=self.figsize)
        gs1 = gridspec.GridSpec(nrows=2, ncols=2, wspace=0.15, height_ratios=[1,.02])
        ax1 = fig.add_subplot(gs1[0, 0], projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(gs1[0, 1], projection=ccrs.PlateCarree())
        ax3 = fig.add_subplot(gs1[1, :], projection=ccrs.PlateCarree())

        # plain white background
        ax = [ax1, ax2, ax3]
        for a in ax:
            a.background_patch.set_visible(False)
            a.outline_patch.set_visible(False)

        # populate left/right panels
        for ax, groupvar in zip([ax1, ax2], ['CD_2010', 'district_final']):
            self._get_single_map(ax, groupvar)

        # populate colorbar
        self._plot_colorbar(ax3, fig)

        # save figure
        if figure_directory is not None:
            utils.make_folder(figure_directory)
            name = '{}_before_after.png'.format(self.state)
            path = figure_directory + name
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        else:
            plt.show()

    def _plot_colorbar(self, ax, fig):
        """Adds colorbar to axis object

        INPUTS:
        ----------------------------------------------------------------------------
        ax : matplotlib axis object

        """
        self.cmap.set_array([])
        bounds = np.linspace(.2, .8, 8, dtype=float).tolist()
        kw = dict(orientation='horizontal', norm=self.norm, spacing='proportional', fraction=1.95, pad=0.5, aspect=30)    
        fig.subplots_adjust(bottom=0.08)
        ax.set_title('Percent Republican', fontsize=12)
        fig.colorbar(self.cmap, ax=ax, **kw)


    def _get_single_map(self, ax, groupvar):
        """
        INPUTS:
        ----------------------------------------------------------------------------
        ax : maplotlib axis object
        groupvar : column in categorical

        """
        xbounds = (self.pcnct_df['INTPTLON10'].min() - .1, self.pcnct_df['INTPTLON10'].max() + .1)
        ybounds = (self.pcnct_df['INTPTLAT10'].min() - .05, self.pcnct_df['INTPTLAT10'].max() + .05)
        ax.set_xlim(xbounds)
        ax.set_ylim(ybounds)

        df_district = self.pcnct_df.dissolve(by=groupvar, aggfunc=np.mean, as_index=True)
        df_district.reset_index(inplace=True)
        df_district = df_district.to_crs(epsg=4267)

        # read in full state shapefiles
        state_df = geo.GeoDataFrame.from_file('../Data-Files/states_shapes/states_shapes.shp')
        state_df = state_df[state_df['NAME'] == states[self.state]]
        state_df = state_df.to_crs(df_district.crs)

        df = geo.overlay(df_district, state_df, how='intersection')
        df = df.dissolve(by=groupvar, aggfunc=np.mean, as_index=True)
        df.reset_index(inplace=True)

        self._get_cmap(df['REP_PCT'], cmap='coolwarm')
        df['color'] = [self.cmap.to_rgba(value) for value in df['REP_PCT'].values]

        for facecolor, state_ in zip(df['color'], df['geometry']):           
                ax.add_geometries([state_], ccrs.PlateCarree(), facecolor=facecolor, edgecolor='black', linewidth=.3)

    def _get_cmap(self, values, cmap, vmin=.3, vmax=.7):
        """
        Normalize and set colormap
        
        Parameters
        ----------
        values : Series or array to be normalized
        cmap : matplotlib Colormap
        normalize : matplotlib.colors.Normalize
        cm : matplotlib.cm
        vmin : Minimum value of colormap. If None, uses min(values).
        vmax : Maximum value of colormap. If None, uses max(values).
        
        Returns
        -------
        n_cmap : mapping of normalized values to colormap (cmap)
        
        """
        mn = vmin or min(values)
        mx = vmax or max(values)
        self.norm = Normalize(vmin=mn, vmax=mx)
        self.cmap = plt.cm.ScalarMappable(norm=self.norm, cmap=cmap)


if __name__ == "__main__":

    # read in precinct data
    for state in ['AL', 'NC', 'PA', 'MD', 'FL', 'CO']:
        with open('{}_data.pkl'.format(state), 'rb') as f:
            ds = pickle.load(f)
            df = ds.pcnct_df
            df = df.to_crs(epsg=4267)
            mm = map_maker(df, state)
            mm.make_maps()