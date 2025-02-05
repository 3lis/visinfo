"""
#####################################################################################################################

    Print a summary of executions in res/

#####################################################################################################################
"""

import  os
import  sys
import  re

res         = "../res"
log         = "log.txt"

versions    = [ "25-02-02_11-42-08" ]
shortcuts   = {
    'ask_img':              'ask_im',
    'ask_share':            'ask_sh',
    'ask_share_strict':     'ask_ss',
    'ask_share_noexplain':  'ask_ne',
    'intro_profile':        'int_pr',
    'profile_conspirator':  'pro_co',
    'profile_moderate':     'pro_mo',
    'profile_rational':     'pro_ra',
    'context':              'ctx___',
    'context_strict':       'ctx_st',
    'reason_base':          'rea_bs',
    'reason_share':         'rea_sh',
    'reason_share_xml':     'rea_sx',
    'reason_share_delimit': 'rea_sd',
}


def get_version ( folder ):
    """
    retrieve the version of the logs based on the folder date
    """
    for v, f in enumerate( versions ):
        if folder < f:
            return v
    return v + 1


def get_info ( lines, version=2 ):
    """
    retrieve essential info
    """

    experiment      = "-----"
    dialogs_pre     = []
    dialogs_post    = []
    model           = "---"
    pre             = ""
    post            = ""
    means           = "---"
    profile         = "---"
    for i, l in enumerate( lines ):
        if "experiment" in l:
            experiment      = l.split()[ -1 ]
        if "model " in l:
            model           = l.split()[ -1 ].replace( "scenario_", '' )
        if "dialogs_pre" in l:
            dialogs_pre     = re.sub( r'[\W]+', ' ', l ).split()[ 1 : ]
        if "dialogs_post" in l:
            dialogs_post    = re.sub( r'[\W]+', ' ', l ).split()[ 1 : ]
        if "News" in l:
            break

    n       = len( lines )
    while True:
        l   = lines[ i ]
        if "mean" in l:
            break
        i   += 1
        if i == n:
            return None
    if version == 0:
        split           = l.split()
        mean_im_tx      = split[ 2 ]
        mean_tx         = split[ 4 ]
        means           = mean_im_tx + 13 * ' ' + mean_tx
    else:
        means           = l[ 6:-1 ]
        means           = means.replace( '  ', ' ' )
    if len( dialogs_pre ):
        for d in dialogs_pre:
            if d in shortcuts.keys():
                pre         += shortcuts[ d ] + ' '
    if len( dialogs_post ):
        for d in dialogs_post:
            if d in shortcuts.keys():
                post        += shortcuts[ d ] + ' '

    return experiment, model, pre, post, means


# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================

if __name__ == '__main__':
    list_res    = sorted( os.listdir( res ) )

    if len( sys.argv ) > 1:             # if there is an argument, use it at the latest result to show
        last_res    = sys.argv[ 1 ]
        if last_res in list_res:
            idx         = list_res.index( last_res )
            list_res    = list_res[ idx : ]

    print( 136 * "_" )
    print( "shortcuts:" )
    for k in shortcuts.keys():
        s   = shortcuts[ k ]
        print( f"{k:<25}: {s}" )
    print( 136 * "_" )
    print()
    header  = "     result     experiment   model"
    header  += 27 * ' '
    header  += "dialogs_pre            dialogs_post"
    header  += 5 * ' '
    header  += "YES+i NO+i  UN+i  YES-i NO-i  UN-i"
    print( header )
    print( 136 * "_" )
    for f in list_res:
        version = get_version( f )
        fname   = os.path.join( res, f, log )
        if not os.path.isfile( fname ):
            print( f"{f}  is not a file" )
            continue
        with open( fname, 'r' ) as fd:
            lines   = fd.readlines()
        if not len( lines ):
            print( f"{f}  has no lines" )
            continue
        info        = get_info ( lines, version=version )
        if info is None:
            print( f"{f}  no info found" )
        else:
            e, m, r, p, v   = info
            if len( m ) > 30:
                m   = m[ : 30 ] + "..."
        print( f"{f}  {e:<6} {m:<33} {r:<22} {p:<17} {v}" )
    print( 136 * "_" )
