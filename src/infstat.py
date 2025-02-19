"""
#####################################################################################################################

    Program for generating statistics over a range of executions in ../res

#####################################################################################################################
"""

import  os
import  sys
import  re
import  json
import  time
import  shutil
import  numpy   as np
import  pandas  as pd
from    statsmodels.formula.api     import ols
from    statsmodels.stats.anova     import anova_lm
from    models                      import models_short_name
import  plot

# specification of the executions to analyze
# if res_range is empty all executions found in ../res are analyzed
# if res_range has only one entry, it is the first execution to process, followed by all the others
# if res_range has two entries, these are the boundaries of the executions to analyze
# if res_range has more than two entries, those entries only are analyzed
res_range           = [ "25-02-05_11-48-20" ]   # this is the first execution with demographics
res_range           = []

res                 = "../res"                  # results directory
dir_json            = "../data"                 # directory with all input data
dir_stat            = "../stat"                 # output directory
f_demo              = "demographics.json"       # filename of demographic data
log                 = "log.txt"                 # filename of execution logs
f_stat              = "stat.txt"                # filename of output statistics
f_plot              = "pl"                      # filename prefix of output plots
frmt_statdir        = "%y-%m-%d-%H-%M"          # datetime format for output directory
columns             = (
    "predia",          # the titles of preliminary dialogs
    "postdia",         # the titles of final part of the dialogs
    "profile",         # the profile induced to the model
    "model",           # the model used
    "gender",          # F/M
    "race",            # race categories as in the demographics file
    "edu",             # education categories as in the demographics file
    "age",             # age categories as in the demographics file
    "politic",         # political affiliation categories as in the demographics file
    "news",            # the news code
    "value",           # true/false
    "yes_img",        # fraction of YES answer for text+image
    "not_img",        # fraction of NO answer for text+image
    "unk_img",        # fraction of missing answer for text+image
    "yes_txt",        # fraction of YES answer for text only
    "not_txt",        # fraction of NO answer for text only
    "unk_txt",        # fraction of missing answer for text only
)

shortcuts   = {
    'ask_img':              'ask_im',
    'ask_share':            'ask_sh',
    'ask_share_strict':     'ask_ss',
    'ask_share_noexplain':  'ask_ne',
    'intro_profile':        'int_pr',
    'profile_conspirator':  'pro_co',
    'profile_moderate':     'pro_mo',
    'profile_rational':     'pro_ra',
    'context':              'context',
    'context_strict':       'ctx_st',
    'reason_3steps':        'rea_3s',
    'reason_base':          'rea_bs',
    'reason_share':         'rea_sh',
    'reason_share_xml':     'rea_sx',
    'reason_share_delimit': 'rea_sd',
}

demography          = None                      # dictionary with demographic categories

def read_demo():
    """
    Read the demographic categories in the json file
    """
    global demography
    dfile               = os.path.join( dir_json, f_demo )
    with open( dfile, 'r' ) as f:
        demography      = json.load( f )


def get_demo( lines ):
    """
    Retrieve demographics info used in the statistics
    """
    gender          = "unspec"
    race            = "unspec"
    edu             = "unspec"
    age             = "unspec"
    politic         = "unspec"

    for l in lines:
        if "political_affiliation" in l:
            politic     = l.lower().split()[ -1 ]
        if "gender" in l:
            if "male" in l:
                gender  = 'M'
            if "female" in l:
                gender  = 'F'
        if "race" in l:
            for i, d in enumerate( demography[ "race" ] ):
                if d in l:
                    race    = f"type{i}"
        if "education" in l:
            for i, d in enumerate( demography[ "education" ] ):
                if d in l:
                    edu     = f"level{i}"
        if "age" in l:
            for i, d in enumerate( demography[ "age" ] ):
                if d in l:
                    age     = f"bin{i}"

    return gender, race, edu, age, politic


def get_predialog( line ):
    """
    Retrieve the pre-dialog settings
    """
    predia      = "unspec"
    profile     = "unspec"
    dialogs     = re.sub( r'[\W]+', ' ', line ).split()[ 1 : ]
    if len( dialogs ) < 2:
        print( f"not enough elements in {dialogs}" )
        return predia, profile
    if "profile_" in dialogs[ 1 ]:
        profile = dialogs[ 1 ].replace( "profile_", "" )
    if len( dialogs ) > 2:
        if dialogs[ 2 ] in shortcuts.keys():
            predia  = shortcuts[ dialogs[ 2 ] ]

    return predia, profile


def get_postdialog( line ):
    """
    Retrieve the post-dialog settings
    """
    postdia     = "unspec"
    dialogs     = re.sub( r'[\W]+', ' ', line ).split()[ 1 : ]
    if not len( dialogs ):
        print( f"not enough elements in {line}" )
        return postdia
    if dialogs[ 0 ] in shortcuts.keys():
        postdia = shortcuts[ dialogs[ 0 ] ]
    if len( dialogs ) > 1:
        if dialogs[ -1 ] in shortcuts.keys():
            postdia  += shortcuts[ dialogs[ -1 ] ][ -3 : ]

    return postdia


def get_info( lines ):
    """
    Retrieve all info used in the statistics
    """
    predia      = "unspec"
    postdia     = "unspec"
    profile     = "unspec"
    model       = "unspec"
    gender      = "unspec"
    race        = "unspec"
    edu         = "unspec"
    age         = "unspec"
    politic     = "unspec"
    news        = []
    value       = []
    yes_img    = []
    not_img    = []
    unk_img    = []
    yes_txt    = []
    not_txt    = []
    unk_txt    = []
    for i, l in enumerate( lines ):
        if "News" in l:
            break
        if "model " in l:
            m_fullname      = l.split()[ -1 ]
            model           = models_short_name[ m_fullname ]
        if "dialogs_pre" in l:
            predia, profile = get_predialog( l )
        if "dialogs_post" in l:
            postdia         = get_postdialog( l )
        if "demographics" in l and not "None" in l:
            demo_lines      = [ l ]
            n   = 1
            l   = lines[ i+n ]
            while l[ : 2 ] == '  ':
                demo_lines.append( l )
                n   += 1
                l   = lines[ i+n ]
                if n > 6:
                    print( "invalid demographic format" )
                    return None
                gender, race, edu, age, politic = get_demo( demo_lines )

    n       = len( lines )
    i       += 1
    while True:
        if "f_mn" in l:
            break
        if "t_mn" in l:
            break
        if "mean" in l:
            break
        l       = lines[ i ]
        v       = l.split()
        if len( v ) != 7:
            print( f"invalid lenght of data: {v}" )
            return None
        news.append( v[ 0 ][ 1 : ] )
        if v[ 0 ][ 0 ] == 'f':
            value.append( 'false' )
        else:
            value.append( 'true' )
        yes_img.append( float( v[ 1 ] ) )
        not_img.append( float( v[ 2 ] ) )
        unk_img.append( float( v[ 3 ] ) )
        yes_txt.append( float( v[ 4 ] ) )
        not_txt.append( float( v[ 5 ] ) )
        unk_txt.append( float( v[ 6 ] ) )
        i       += 1
        if i == n:
            print( "missing end of news results" )
            return None

    news        = np.array( news )
    value       = np.array( value )
    yes_img    = np.array( yes_img )
    not_img    = np.array( not_img )
    unk_img    = np.array( unk_img )
    yes_txt    = np.array( yes_txt )
    not_txt    = np.array( not_txt )
    unk_txt    = np.array( unk_txt )
    predia      = np.full( news.shape, predia )
    postdia     = np.full( news.shape, postdia )
    profile     = np.full( news.shape, profile )
    model       = np.full( news.shape, model )
    gender      = np.full( news.shape, gender )
    race        = np.full( news.shape, race )
    edu         = np.full( news.shape, edu )
    age         = np.full( news.shape, age )
    politic     = np.full( news.shape, politic )

    data            = {
        "predia":          predia,
        "postdia":         postdia,
        "profile":         profile,
        "model":           model,
        "gender":          gender,
        "race":            race,
        "edu":             edu,
        "age":             age,
        "politic":         politic,
        "news":            news,
        "value":           value,
        "yes_img":        yes_img,
        "not_img":        not_img,
        "unk_img":        unk_img,
        "yes_txt":        yes_txt,
        "not_txt":        not_txt,
        "unk_txt":        unk_txt,
    }

    return data


def collect_data():
    """
    Scan the results, collecting all data

    return:             [pandas.core.frame.DataFrame] the data in pandas DataFrame
    """

    list_res    = sorted( os.listdir( res ) )
    match len( res_range ):
        case 1:
            first   = res_range[ 0 ]
            assert first in list_res, f"first specified result {first} not found"
            i_first     = list_res.index( first )
            list_res    = list_res[ i_first : ]
        case 2:
            first   = res_range[ 0 ]
            last    = res_range[ -1 ]
            assert first in list_res, f"first specified result {first} not found"
            assert last in list_res, f"last specified result {last} not found"
            i_first     = list_res.index( first )
            i_last      = list_res.index( last )
            list_res    = list_res[ i_first : i_last+1 ]
        case _:
            if len( res_range ):
                list_res    = res_range

    arrays  = dict()
    for c in columns:
        arrays[ c ] = []

    for f in list_res:                          # scan all selected results
        fname   = os.path.join( res, f, log )
        if not os.path.isfile( fname ):
            print( f"{f}  is not a file" )
            continue
        with open( fname, 'r' ) as fd:
            lines   = fd.readlines()
        if not len( lines ):
            print( f"{f}  has no lines" )
            continue
        data        = get_info ( lines )        # get data for one execution
        if data is None:
            print( f"{f}  no info found" )
            continue
        for c in columns:                       # accumulate data
            arrays[ c ].append( data[ c ] )
        print( f"{f}  done" )

    for c in columns:
        v           = arrays[ c ]
        arrays[ c ] = np.concatenate( v )

    df          = pd.DataFrame( arrays )
    return df


def means( df ):
    """
    Compute means of the main scores grouped by the main variables independently.
    Use the basic grouping and aggregation functions of pandas, and aggregate by models
    and all the other variables.

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame

    return:             [tuple] means for profile, age, gender, race, edu, politic (all models), and same by models
    """
    scores      = [ 'yes_img', 'yes_txt' ]

    mm          = df.groupby( [ 'value', 'profile' ] )[ scores ].mean()
    ma          = df.groupby( [ 'value', 'age' ] )[ scores ].mean()
    mg          = df.groupby( [ 'value', 'gender' ] )[ scores ].mean()
    mr          = df.groupby( [ 'value', 'race' ] )[ scores ].mean()
    me          = df.groupby( [ 'value', 'edu' ] )[ scores ].mean()
    mp          = df.groupby( [ 'value', 'politic' ] )[ scores ].mean()
    mmm         = df.groupby( [ 'model', 'value', 'profile' ] )[ scores ].mean()
    mma         = df.groupby( [ 'model', 'value', 'age' ] )[ scores ].mean()
    mmg         = df.groupby( [ 'model', 'value', 'gender' ] )[ scores ].mean()
    mmr         = df.groupby( [ 'model', 'value', 'race' ] )[ scores ].mean()
    mme         = df.groupby( [ 'model', 'value', 'edu' ] )[ scores ].mean()
    mmp         = df.groupby( [ 'model', 'value', 'politic' ] )[ scores ].mean()

    return mm, ma, mg, mr, me, mp, mmm, mma, mmg, mmr, mme, mmp


def anova_1( df, score ):
    """
    Compute one-way anova of one the scores for all variables.
    Construct the regression formula in the 'R'-style used in the ols function of statsmodels,
    then apply the anova_lm to the regression model returned by ols
    See:
        https://www.statsmodels.org/dev/generated/statsmodels.formula.api.ols.html
        https://www.statsmodels.org/dev/generated/statsmodels.stats.anova.anova_lm.html

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        score           [str] one of the response scores

    return:             [tuple] anova for profile, age, gender, race, edu, politic
    """
    formula     = f"{score} ~ C(profile)"
    model       = ols( formula, df ).fit()
    am          = anova_lm( model )
    formula     = f"{score} ~ C(age)"
    model       = ols( formula, df ).fit()
    aa          = anova_lm( model )
    formula     = f"{score} ~ C(gender)"
    model       = ols( formula, df ).fit()
    ag          = anova_lm( model )
    formula     = f"{score} ~ C(race)"
    model       = ols( formula, df ).fit()
    ar          = anova_lm( model )
    formula     = f"{score} ~ C(edu)"
    model       = ols( formula, df ).fit()
    ae          = anova_lm( model )
    formula     = f"{score} ~ C(politic)"
    model       = ols( formula, df ).fit()
    ap          = anova_lm( model )

    return am, aa, ag, ar, ae, ap


def do_plots( df ):
    """
    Do various plots
    """
    fname       = os.path.join( dir_stat, f_plot )
    plot.plot_models( df, groups=[ "value", "age" ], fname=fname )
    plot.plot_models( df, groups=[ "value", "gender" ], fname=fname )
    plot.plot_models( df, groups=[ "value", "race" ], fname=fname )
    plot.plot_models( df, groups=[ "value", "edu" ], fname=fname )
    plot.plot_models( df, groups=[ "value", "politic" ], fname=fname )
    plot.plot_models( df, groups=[ "value", "profile" ], fname=fname )
    plot.plot_models( df, groups=[ "value", "predia" ], fname=fname )
    plot.plot_models( df, groups=[ "value", "postdia" ], fname=fname )
    unk = [ "unk_img", "unk_txt" ]
    plot.plot_models( df, groups=[ "value", "predia" ], values=unk, fname=fname+"_unk" )
    plot.plot_models( df, groups=[ "value", "postdia" ], values=unk, fname=fname+"_unk" )


def print_means( f, m ):
    """
    Print mean values
    """
    mm, ma, mg, mr, me, mp = m
    f.write( 80 * "=" + "\n" )
    f.write( " means ".center( 80, ' ' ) + '\n' )
    f.write( 80 * "=" + "\n\n" )
    f.write( " profile ".center( 80, '=' ) + '\n' )
    f.write( mm.to_string() + '\n\n' )
    f.write( " age ".center( 80, '=' ) + '\n' )
    f.write( ma.to_string() + '\n\n' )
    f.write( " gender ".center( 80, '=' ) + '\n\n' )
    f.write( mg.to_string() + '\n\n' )
    f.write( " race ".center( 80, '=' ) + '\n' )
    f.write( mr.to_string() + '\n\n' )
    f.write( " education ".center( 80, '=' ) + '\n\n' )
    f.write( me.to_string() + '\n\n' )
    f.write( " politic ".center( 80, '=' ) + '\n\n' )
    f.write( mp.to_string() + '\n\n' )
    f.write( 80 * "=" + "\n\n" )


def print_anova_1( f, a, scores ):
    """
    Print one-way anova
    """
    f.write( 80 * "=" + "\n" )
    f.write( " one-way anova ".center( 80, ' ' ) + '\n' )
    f.write( 80 * "=" + "\n\n" )
    f.write( " profile ".center( 80, '=' ) + '\n' )
    for s in scores:
        f.write( s.center( 80, '-' ) + '\n' )
        f.write( a[ s ][ 0 ].to_string() + '\n\n' )
    f.write( " age ".center( 80, '=' ) + '\n' )
    for s in scores:
        f.write( s.center( 80, '-' ) + '\n' )
        f.write( a[ s ][ 1 ].to_string() + '\n\n' )
    f.write( " gender ".center( 80, '=' ) + '\n' )
    for s in scores:
        f.write( s.center( 80, '-' ) + '\n' )
        f.write( a[ s ][ 2 ].to_string() + '\n\n' )
    f.write( " race ".center( 80, '=' ) + '\n' )
    for s in scores:
        f.write( s.center( 80, '-' ) + '\n' )
        f.write( a[ s ][ 3 ].to_string() + '\n\n' )
    f.write( " education ".center( 80, '=' ) + '\n' )
    for s in scores:
        f.write( s.center( 80, '-' ) + '\n' )
        f.write( a[ s ][ 4 ].to_string() + '\n\n' )
    f.write( " politic ".center( 80, '=' ) + '\n' )
    for s in scores:
        f.write( s.center( 80, '-' ) + '\n' )
        f.write( a[ s ][ 5 ].to_string() + '\n\n' )
    f.write( 80 * "=" + "\n\n" )


def do_stat( df ):
    """
    Do basic statistics and write it on file
    """
    scores      = [ 'yes_img', 'yes_txt' ]
    models      = list( df[ "model" ].unique() )
    fname       = os.path.join( dir_stat, f_stat )
    f           = open( fname, 'w' )
    f.write( 80 * "=" + "\n\n" )
    all_means   = means( df )
    f.write( 80 * "+" + "\n" )
    f.write( "all models".center( 80, ' ' ) + '\n' )
    f.write( 80 * "+" + "\n\n" )
    print_means( f, all_means[ : 6 ] )
    f.write( "\n\n" )
    f.write( 80 * "+" + "\n" )
    f.write( "by model".center( 80, ' ' ) + '\n' )
    f.write( 80 * "+" + "\n\n" )
    print_means( f, all_means[ 6 : ] )
    anova                   = {}
    f.write( "\n\n" )
    f.write( 80 * "+" + "\n" )
    f.write( "all models".center( 80, ' ' ) + '\n' )
    f.write( 80 * "+" + "\n\n" )
    for s in scores:
        anova[ s ]          = anova_1( df, s )
    print_anova_1 ( f, anova, scores )
    for m in models:
        f.write( "\n\n" )
        f.write( 80 * "+" + "\n" )
        f.write( m.center( 80, ' ' ) + '\n' )
        f.write( 80 * "+" + "\n\n" )
        for s in scores:
            anova[ s ]          = anova_1( df[ (df[ 'model' ]==m ) ], s )
        print_anova_1 ( f, anova, scores )
    f.close()



# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================
if __name__ == '__main__':
    # first, it is necessary to read the demographics from json
    read_demo()
    df          = collect_data()        # all data in pandas DataFrame
    now_time    = time.strftime( frmt_statdir )
    dir_stat    = os.path.join( dir_stat, now_time )
    if not os.path.isdir( dir_stat ):
        os.makedirs( dir_stat )
    # save a copy of this file and the plotting script
    shutil.copy( "infstat.py", dir_stat )
    shutil.copy( "plot.py", dir_stat )
    do_plots( df )
    do_stat( df )
