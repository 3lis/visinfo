"""
#####################################################################################################################

    Module to save results of executions

#####################################################################################################################
"""

import  os
import  sys
import  platform
import  pickle
import  csv
import  numpy       as np

cnfg                = None                  # parameter obj assigned by main_exec.py


# ===================================================================================================================
#
#   Functions to write the results on pickle file and compute stats on it
#   - write_pickle
#   - get_pickle
#   - write_stats
#
# ===================================================================================================================

def write_pickle( fname, results ):
    """
    Save raw results in a pickle file

    params:
        fname       [str] pickle file with path and extension
        results     [dict] of scores per news
    """
    with open( fname, 'wb' ) as f:
        pickle.dump( ( results ), f )


def get_pickle( fname ):
    """
    Get content of pickle file

    params:
        fname       [str] pickle file with path and extension
    """
    with open( fname, 'rb' ) as f:
        return pickle.load( f )


def write_stats( fcsv, fpkl=None, results=None ):
    """
    Write in CSV file stats about the results, either from the pickle file or from the data passed

    params:
        fcsv        [str] csv file with path and extension
        fpkl        [str] optional pickle file with path and extension
        results     [dict] optional structure with scores per news
    """
    if fpkl is not None:
        results     = get_pickle( fpkl )
    else:
        assert results is not None, "ERROR: no pickle file or dict of results found"

    csv_header  = [ "News" ]
    csv_rows    = []

    # stats for executions using news with AND without images
    if "with_img" in results and "no_img" in results:
        csv_header      += [ "YES (img+txt)", "YES (txt)" ]
        all_res_img     = results[ "with_img" ]
        all_res_txt     = results[ "no_img" ]
        res_img         = []
        res_txt         = []
        k_items         = sorted( list( all_res_img.keys() ) )
        for k  in k_items:
            ri          = all_res_img[ k ]
            rt          = all_res_txt[ k ]
            n           = len( ri )        # equal to the num of completions
            assert n > 0, "ERROR: no completions for news {k} in write_stats()"
            vi          = ri.sum() / n
            vt          = rt.sum() / n
            res_img.append( vi )
            res_txt.append( vt )
            csv_rows.append( [ k, f"{vi:.3f}", f"{vt:.3f}" ] )

        res_img     = np.array( res_img )
        res_txt     = np.array( res_txt )
        mean_img    = res_img.mean()
        mean_txt    = res_txt.mean()
        std_img     = res_img.std()
        std_txt     = res_txt.std()
        csv_rows.append( [  "mean [std]",
                            f"{mean_img:.3f} [{std_img:.3f}]",
                            f"{mean_txt:.3f} [{std_txt:.3f}]"
        ] )

    # stats for executions using news with OR without images
    else:
        csv_header  += [ "Fraction of YES" ]
        res         = []
        k_items     = sorted( list( results.keys() ) )
        for k  in k_items:
            rs      = results[ k ]
            n       = len( rs )    # equal to the num of completions
            assert n > 0, "ERROR: no completions for news {k} in write_stats()"
            r       = rs.sum() / n
            res.append( r )
            csv_rows.append( [ k, f"{r:.3f}" ] )

        res     = np.array( res )
        mean    = res.mean()
        std     = res.std()
        csv_rows.append( [ "mean [std]", f"{mean:.3f} [{std:.3f}]" ] )

    with open( fcsv, mode='w', newline='' ) as f:
        w   = csv.writer( f )
        w.writerow( csv_header )
        w.writerows( csv_rows )



# ===================================================================================================================
#
#   Functions to write the results on textual log file
#   - write_header
#   - write_dialog
#   - write_dialogs
#   - write_all
#
# ===================================================================================================================

def write_header( fstream ):
    """
    Write the initial part of the log file with info about the execution command and parameters

    params:
        fstream     [TextIOWrapper] text stream of the output file
    """
    # write the command that executed the program
    command     = sys.executable + " " + " ".join( sys.argv )
    host        = platform.node()
    fstream.write( 60 * "=" + "\n\n" )
    fstream.write( "executing:\n" + command )
    fstream.write( "\non host " + host + "\n\n" )

    # write info on config parameters
    fstream.write( 60 * "=" + "\n\n" )
    fstream.write( str( cnfg ) )
    fstream.write( "\n" + 60 * "=" + "\n\n" )


def write_dialog( fstream, prompt, completions, mode="chat" ):
    """
    Write the content of prompts and completions on the log file

    params:
        fstream     [TextIOWrapper] text stream of the output file
        prompt      [list] of dialog messages
        completions [list] of [str]
        mode        [str] "cmpl" or "chat"
    """
    if mode == "cmpl":
        fstream.write( f"PROMPT:\n{prompt}\n\n" )
    elif mode == "chat":
        for p in prompt:
            fstream.write( f"ROLE: {p['role']}\n" )
            c   = p[ 'content' ]
            if not isinstance( c, str ):
                assert c[ 0 ][ 'type' ] == 'text', f"ERROR: unrecognized structure of completion '{c}'"
                c   = c[ 0 ][ 'text' ]
            fstream.write( f"PROMPT:\n{c}\n\n" )
    else:
        print( f"ERROR: mode '{mode}' not supported" )
        sys.exit()

    for i, c in enumerate( completions ):
        fstream.write( 60 * "-" + "\n\n" )
        fstream.write( f"COMPLETION #{i}:\n{c}\n\n" )


def write_dialogs( fstream, prompts, completions, results, img_names, mode="chat" ):
    """
    Write the log of all dialogs

    params:
        fstream     [TextIOWrapper] text stream of the output file
        prompt      [list] of prompts
        completions [list] of completions
        results     [dict] of scores per news
        img_names   [list] of image names
        mode        [str] "cmpl" or "chat"
    """
    fstream.write( "\n" + 60 * "=" + "\n" )
    news_list   = cnfg.news_ids
    if cnfg.experiment == "both":
        news_list   += cnfg.news_ids
    for i, pr, compl, name in zip( news_list, prompts, completions, img_names ):
        if len( name ):
            fstream.write( f"\n-------------- News {i} with image {name} ---------------\n\n" )
        else:
            fstream.write( f"\n---------------- News {i} with no image -------------------\n\n" )
        write_dialog( fstream, pr, compl, mode=mode )
        fstream.write( 60 * "=" + "\n" )


def write_all( fstream, prompts, completions, results, img_names, fcsv, fpkl, mode="chat" ):
    """
    Write all result files (text log, csv, pkl)

    params:
        fstream     [TextIOWrapper] text stream of the output file
        prompt      [list] of prompts
        completions [list] of completions
        results     [dict] of scores per news
        img_names   [list] of image names
        fcsv        [str] csv file with path and extension
        fpkl        [str] pickle file with path and extension
        mode        [str] "cmpl" or "chat"
    """
    write_pickle( fpkl, results )
    write_stats( fcsv, results=results )
    write_header( fstream )

    # unix command to pretty print the csv in the text log
    cmd     = f"column -s, -t <{fcsv}"
    with os.popen( cmd ) as r:
        s   = r.read()
    fstream.write( s )

    write_dialogs( fstream, prompts, completions, results, img_names, mode=mode )
