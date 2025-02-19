"""
#####################################################################################################################

    Module for plotting

#####################################################################################################################
"""

import  os
import  sys
import  copy
import  pickle
import  numpy               as np
import  pandas              as pd
from    matplotlib          import pyplot
from    matplotlib.patches  import Patch, Polygon
from    matplotlib.lines    import Line2D

figsize         = ( 18.0, 8.0 )                             # figure size in inches
labelspacing    = 1.2
extension       = ".pdf"

# generic set of colors, grouped 4x4
# therefore in case of single group, better to use every 4 color,
# and in case of group of 2, every 2 color
dim_colors  = (
        '#e88030',
        '#ea3030',
        '#eac130',
        '#ed6ecb',
        '#129dad',
        '#124dfd',
        '#1740ed',
        '#19bd9d',
        '#70a3ff',
        '#5d8500',
        '#62cf02',
        '#00c28b',
        '#00f25b',
        '#00b29b',
        '#00a2bb',
        '#af907f',
        '#afb01f',
        '#af509f',
        '#a0907f',
)
# markers
dim_markers = ( 'o',  '*',  'D',  'P',  'X',  'h', 'H', '<', '>', 'x' )

char_len    = 0.009                                     # typical length of a character in legend, in plot units



def plot_values( df, groups=["value","age"], values=["yes_img", "yes_txt"], fname="plot", suptitle='' ):
    """
    Generate one plot of the specified values, for a specified group of independent variables

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame
        groups      [list] columns of independent variables
        values      [list] the numerical values to be plot
        fname       [str] name of the output file
        suptitle    [str] plot title
    """

    columns = df.columns
    y       = []                        # list of all vectors to plot
    labels  = []                        # labels for all combinations of independent variables
    cat     = dict()                    # all categorical entries found for independent variables
    comb    = dict()                    # vectors for all combinations of independent variables
    for g in groups:                    # gather the categorical entries for independent variables
        assert g in columns, f"there is no column named {g}"
        cat[ g ]    = list( df[ g ].unique() )

    for g in groups:                    # collect all vectors for combinations of independent variables
        if not len( comb ):
            first           = True
        else:
            first           = False
            new_comb        = dict()
        for c in cat[ g ]:
            if first:
                comb[ c ]   = df[ df[ g ] == c ]
            else:
                for prev in comb.keys():
                    dframe  = comb[ prev ]
                    new     = prev + '-' + c
                    new_comb[ new ] = dframe[ dframe[ g ] == c ]
        if not first:
            comb            = new_comb

    for c in comb.keys():               # assign vectors to y and their key combinations to labels
        dframe  = comb[ c ]
        for v in values:
            y.append( dframe [ v ] )
            labels.append( f"{v} for {c}" )

    n           = len( y )
    ng          = len( df[ groups[ 0 ] ].unique() ) * len( values )
    # compute a proper separation of bars by group
    x           = [  i + i // ng for i in range( n ) ]
    step        = 1. / ( x[ -1 ] + 1 )
    match ng:                           # see comments in the definition of dim_colors
        case 1:
            colors      = dim_colors[ 0:4*n:4 ]
        case 2:
            colors      = dim_colors[ 0:2*n:2 ]
        case _:
            colors      = dim_colors[ : n ]

    pyplot.rcParams.update( { "font.size": 14 } )
    handles     = [ Patch( facecolor=c, label=l ) for c,l in zip( colors, labels ) ]
    fig, ax     = pyplot.subplots( figsize=figsize )

    for i, ( xx, data ) in enumerate( zip( x, y ) ):
        m   = data.mean()
        s   = data.std() / 2.
        ax.bar( xx, m, yerr=s, color=colors[ i ], linewidth=2 )
        ax.set_xticks( [] )

    ax.set_ylabel( "YES fraction" )
    ax.set_ylim( bottom=0.0, top=1.0 )

    # compute an estimate of the max occupancy of labels in the plot
    lab_len     = char_len * max( [ len( l ) for l in labels ] )
    # compute the current geometry of the plotting box
    box = ax.get_position()
    # and now make room on the right for the labels
    ax.set_position([box.x0, box.y0, box.width * ( 1 - lab_len ), box.height])
    ax.legend( handles=handles, loc="center", bbox_to_anchor=(1+lab_len, 0.5), labelspacing=labelspacing )
    fname       = fname + extension
    pyplot.suptitle( suptitle, x=0.4 )
    pyplot.savefig( fname, bbox_inches="tight", pad_inches=0 )
    pyplot.close()
    pyplot.clf()


def plot_models( df, groups=["value","age"], values=["yes_img", "yes_txt"], fname="pl" ):
    """
    Wrapper for executing plot_values() on all models found in the dataset, together with a final
    plot with all models data

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame
        groups      [list] columns of independent variables
        values      [list] the numerical values to be plot
        fname       [str] name of the output file
    """
    models      = list( df[ "model" ].unique() )    # find all models used in the dataframe

    g           = groups[ -1 ]
    for m in models:
        name    = f"{fname}_{m}_{g}"
        plot_values( df[ df[ "model" ] == m ], groups=groups, values=values, fname=name, suptitle=f"model {m}" )
    name    = f"{fname}_all_{g}"
    plot_values( df, groups=groups, values=values, fname=name, suptitle="all models" )
