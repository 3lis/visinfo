"""
#####################################################################################################################

    Module to save results of executions

#####################################################################################################################
"""

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


def write_stats( fpkl, fcsv ):
    """
    Write in CSV file stats about the raw results in the pickle file

    params:
        fpkl        [str] pickle file with path and extension
        fcsv        [str] csv file with path and extension
    """
    results     = get_pickle( fpkl )
    csv_header  = [ "News" ]
    csv_rows    = []

    # stats for executions using news with AND without images
    if "with_img" in results and "no_img" in results:
        csv_header      += [ "Fraction of YES (img+txt)", "Fraction of YES (txt)" ]
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
        fstream.write( f"\nmean: {mean:5.2f}\tstd: {std:5.2f}\n" )
        csv_rows.append( [  "mean [std]", f"{mean:.3f} [{std:.3f}]" ] )

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


def write_dialog( fstream, prompt, completions ):
    """
    Write the content of prompts and completions on the log file

    params:
        fstream     [TextIOWrapper] text stream of the output file
        prompt      [list] of dialog messages
        completions [list] of [str]
    """
    for p in prompt:
        fstream.write( f"ROLE: {p['role']}\n" )
        fstream.write( f"PROMPT:\n{p['content']}\n\n" )

    fstream.write( 60 * "-" + "\n\n" )

    for i, c in enumerate( completions ):
        fstream.write( f"COMPLETION #{i}:\n{c}\n\n" )
        # fstream.write( f"COMPLETION #{i:3d}:\n{c}\n\n" )


def write_dialogs( fstream, prompts, completions, results, img_names ):
    """
    Write the log of all dialogs

    params:
        fstream     [TextIOWrapper] text stream of the output file
        prompt      [list] of prompts
        completions [list] of completions
        results     [dict] of scores per news
        img_names   [list] of image names
    """
    # question        = cnfg.dialogs[ -1 ]    # the stats are about only the last prompt question
    # fstream.write( f"Results for question {question}:\n" )
    # fstream.write( '\nItem\t\tFraction of "yes"\n' )
    #
    # # stats for executions using news with AND without images
    # if cnfg.experiment == "both":
    #     all_res_img     = []
    #     all_res_noi     = []
    #     fstream.write( '\t\twith image\twithout\n' )
    #     res_img         = results[ "with_img" ]
    #     res_noi         = results[ "no_img" ]
    #     k_items         = list( res_img.keys() )
    #     k_items.sort()
    #     for k  in k_items:
    #         r_i         = res_img[ k ]
    #         r_n         = res_noi[ k ]
    #         n           = len( r_n )
    #         if not n:
    #             fstream.write( f"{k:>3}\t\t0.00\t\t0.00\n" )
    #         else:
    #             yi      = r_i.sum() / n
    #             yn      = r_n.sum() / n
    #             all_res_img.append( yi )
    #             all_res_noi.append( yn )
    #             fstream.write( f"{k:>3}\t\t{yi:4.2f}\t\t{yn:4.2f}\n" )
    #
    #     all_res_img     = np.array( all_res_img )
    #     mean_img        = all_res_img.mean()
    #     std_img         = all_res_img.std()
    #     all_res_noi     = np.array( all_res_noi )
    #     mean_noi        = all_res_noi.mean()
    #     std_noi         = all_res_noi.std()
    #     fstream.write( f"\nmean [std]:\t{mean_img:4.2f} [{std_img:4.2f}]\t{mean_noi:4.2f} [{std_noi:4.2f}]\n" )
    #
    # # stats for executions using news with OR without images
    # else:
    #     all_res         = []
    #     k_items         = list( results.keys() )
    #     k_items.sort()
    #     for k  in k_items:
    #         res         = results[ k ]
    #         n           = len( res )
    #         if not n:
    #             fstream.write( f"{k:>3}\t\t0.00\n" )
    #         else:
    #             r       = res.sum() / n
    #             all_res.append( r )
    #             fstream.write( f"{k:>3}\t\t{r:4.2f}\n" )
    #
    #     all_res         = np.array( all_res )
    #     mean            = all_res.mean()
    #     std             = all_res.std()
    #     fstream.write( f"\nmean: {mean:5.2f}\tstd: {std:5.2f}\n" )

    # log of all dialogs
    fstream.write( "\n" + 60 * "=" + "\n" )
    news_list   = cnfg.news_ids
    if cnfg.experiment == "both":
        news_list   += cnfg.news_ids
    for i, pr, compl, name in zip( news_list, prompts, completions, img_names ):
        if len( name ):
            fstream.write( f"\n---------------- News {i} with image {name} -----------------\n\n" )
        else:
            fstream.write( f"\n------------------- News {i} with no image -------------------\n\n" )
        write_dialog( fstream, pr, compl )
        fstream.write( 60 * "=" + "\n" )
