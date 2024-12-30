"""
#####################################################################################################################

    Main file to execute the program

    For help
        $ python main_exec.py -h

#####################################################################################################################
"""

import  os
import  sys
import  shutil
import  platform
import  time
import  pickle
import  numpy       as np

import  load_cnfg                               # this module sets program parameters
import  prompt      as prmpt                    # this module composes the prompts
import  complete    as cmplt                    # this module performs LLM completions

# this module lists the available LLMs
from    models      import models, models_endpoint, models_interface

# execution directives
DO_NOTHING              = False                 # for debugging
DEBUG                   = False                 # temporary specific debugging

frmt_response           = "%y-%m-%d_%H-%M-%S"   # datetime format for filenames
now_time                = None                  # the global current datetime
dir_res                 = '../res'              # folder of results
dir_json                = '../data'             # folder of json data

cnfg                    = None                  # object containing the execution configuration (see load_cnfg.py)

# globals pointing to current execution folders and files
# NOTE they will be validated in init_dirs()
exec_dir                = None
exec_src                = 'src'
exec_data               = 'data'
exec_log                = 'run.log'
exec_file               = 'res.pkl'


# ===================================================================================================================
#
#   Utilities to set up execution
#   - init_dirs
#   - init_cnfg
#   - archive
#
# ===================================================================================================================

def init_dirs():
    """
    Set paths and create directories where to save the current execution
    """
    global exec_dir, exec_src, exec_data        # dirs
    global exec_log, exec_file                  # files

    exec_dir     = os.path.join( dir_res, now_time )
    while os.path.isdir( exec_dir ):
        if cnfg.VERBOSE:
            print( f"WARNING: a folder with the timestamp {exec_dir} already exists." )
            print( "Creating a folder with a timestamp a second ahead.\n" )
        sec         = int( exec_dir[ -2: ] )
        sec         += 1
        exec_dir    = f"{exec_dir[ :-2 ]}{sec:02d}"

    exec_src        = os.path.join( exec_dir, exec_src )
    exec_data       = os.path.join( exec_dir, exec_data )

    os.makedirs( exec_dir )
    os.makedirs( exec_src )
    os.makedirs( exec_data )
    exec_log        = os.path.join( exec_dir, exec_log )
    exec_file       = os.path.join( exec_dir, exec_file )


def init_cnfg():
    """
    Set execution parameters received from command line and python configuration file
    NOTE Execute this function before init_dirs()
    """
    global cnfg
    global now_time

    cnfg            = load_cnfg.Config()                    # instantiate the configuration object

    # load parameters from command line
    line_kwargs     = load_cnfg.read_args()                 # read the arguments in the command line
    cnfg.load_from_line( line_kwargs )                      # and parse their value into the configuration obj

    # load parameters from configuration file
    if cnfg.CONFIG is not None:
        exec( "import " + cnfg.CONFIG )                     # exec the import statement
        file_kwargs     = eval( cnfg.CONFIG + ".kwargs" )   # assign the content to a variable
        cnfg.load_from_file( file_kwargs )                  # read the configuration file

    else:                                                   # default configuration
        cnfg.model_id       = 0                             # use the defaul model
        cnfg.n_returns      = 1                             # just one response
        cnfg.max_tokens     = 50                            # afew tokens
        cnfg.top_p          = 1                             # set a reasonable default
        cnfg.temperature    = 0.3                           # set a reasonable default
        cnfg.dialogs        = []                            # set a reasonable default
        cnfg.news_ids       = []                            # set a reasonable default

    if not hasattr( cnfg, 'experiment' ):
        cnfg.experiment         = None                      # whether experiment uses images or not

    # overwrite command line arguments
    if cnfg.DIALOG is not None:         cnfg.dialogs    = cnfg.DIALOG
    if cnfg.MAXTOKENS is not None:      cnfg.max_tokens = cnfg.MAXTOKENS
    if cnfg.MODEL is not None:          cnfg.model_id   = cnfg.MODEL
    if cnfg.NRETURNS is not None:       cnfg.n_returns  = cnfg.NRETURNS

    # if a model is used, from its index derive the complete model name and usage mode
    if hasattr( cnfg, 'model_id' ):
        assert cnfg.model_id < len( models ), f"error: model # {cnfg.model_id} not available"
        cnfg.model          = models[ cnfg.model_id ]
        cnfg.mode           = models_endpoint[ cnfg.model ]
        cnfg.interface      = models_interface[ cnfg.model ]

    now_time        = time.strftime( frmt_response )    # string used for composing file names of results

    # export information from config
    if hasattr( cnfg, 'f_dialog' ):     prmpt.f_dialog  = cnfg.f_dialog
    if hasattr( cnfg, 'detail' ):       prmpt.detail    = cnfg.detail

    # pass the configuration object to module complete.py
    cmplt.cnfg       = cnfg


def archive():
    """
    Save a copy of current python source and json data files in the execution folder
    """
    jfiles  = ( "dialogs.json",
                "news.json" )

    pfiles  = [ "main_exec.py",
                "prompt.py",
                "load_cnfg.py",
                "models.py",
                "complete.py" ]

    if cnfg.CONFIG is not None:
        pfiles.append( cnfg.CONFIG + ".py" )

    for pfile in pfiles:
        shutil.copy( pfile, exec_src )
    for jfile in jfiles:
        jfile   = os.path.join( dir_json, jfile )
        shutil.copy( jfile, exec_data )



# ===================================================================================================================
#
#   Functions to write the results on file
#   - write_header
#   - write_content
#   - write_results
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


def write_content( fstream, prompt, completions ):
    """
    Write the content of prompts and completions on the log file

    params:
        fstream     [TextIOWrapper] text stream of the output file
        prompt      [list] of dialog messages
        completions [list] of [str]
    """
    for p in prompt:
        fstream.write( f"ROLE: {p['role']}\n" )
        fstream.write( f"PROMPT:\n {p['content']}\n\n" )

    for i, c in enumerate( completions ):
        fstream.write( f"COMPLETION #{i}:\n{c}\n\n" )
        # fstream.write( f"COMPLETION #{i:3d}:\n{c}\n\n" )


def write_results( fstream, prompts, completions, results, img_names ):
    """
    Write stats about the results and then the log of all dialogs

    params:
        fstream     [TextIOWrapper] text stream of the output file
        prompt      [list] of prompts
        completions [list] of completions
        results     [dict] of scores per news
        img_names   [list] of image names
    """
    question        = cnfg.dialogs[ -1 ]    # the stats are about only the last prompt question
    all_n           = 0
    fstream.write( f"Results for question {question}:\n" )
    fstream.write( '\nItem\t\tFraction of "yes"\n' )

    # stats for executions using news with AND without images
    if cnfg.experiment == "both":
        all_res_img     = []
        all_res_noi     = []
        fstream.write( '\t\twith image\twithout\n' )
        res_img         = results[ "with_img" ]
        res_noi         = results[ "no_img" ]
        k_items         = list( res_img.keys() )
        k_items.sort()
        for k  in k_items:
            r_i         = res_img[ k ]
            r_n         = res_noi[ k ]
            n           = len( r_n )
            if not n:
                fstream.write( f"{k:>3}\t\t0.00\t\t0.00\n" )
            else:
                yi      = r_i.sum() / n
                yn      = r_n.sum() / n
                all_n   += n
                all_res_img.append( yi )
                all_res_noi.append( yn )
                fstream.write( f"{k:>3}\t\t{yi:4.2f}\t\t{yn:4.2f}\n" )

        all_res_img     = np.array( all_res_img )
        mean_img        = all_res_img.mean()
        std_img         = all_res_img.std()
        all_res_noi     = np.array( all_res_noi )
        mean_noi        = all_res_noi.mean()
        std_noi         = all_res_noi.std()
        fstream.write( f"\nmean [std]:\t{mean_img:4.2f} [{std_img:4.2f}]\t{mean_noi:4.2f} [{std_noi:4.2f}]\n" )

    # stats for executions using news with OR without images
    else:
        all_res         = []
        k_items         = list( results.keys() )
        k_items.sort()
        for k  in k_items:
            res         = results[ k ]
            n           = len( res )
            if not n:
                fstream.write( f"{k:>3}\t\t0.00\n" )
            else:
                r       = res.sum() / n
                all_n   += n
                all_res.append( r )
                fstream.write( f"{k:>3}\t\t{r:4.2f}\n" )

        all_res         = np.array( all_res )
        mean            = all_res.mean()
        std             = all_res.std()
        fstream.write( f"\nmean: {mean:5.2f}\tstd: {std:5.2f}\n" )

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
        write_content( fstream, pr, compl )
        fstream.write( 60 * "=" + "\n" )



# ===================================================================================================================
#
#   Main functions calling the LLM on news data
#   - check_reply
#   - ask_news
#   - do_exec
#
# ===================================================================================================================

def check_reply( completion ):
    """
    Check the model answers in response to yes/no questions.

    params:
        completion  [list] of completion text

    return:         [np.array] of booleans
    """
    nc      = len( completion )
    res     = np.full( nc, False )

    for i, c in enumerate( completion ):
        c   = c.lower()
        if "yes" in c:              # this check is not very robust, but okay for now...
            res[ i ]    = True

    return res


def ask_news( with_img=True ):
    """
    Prepare the prompts and obtain the model completions

    params:
        with_img    [bool] whether the prompts include image and text

    return:
        [tuple] of:
                    prompts     [list] of all prompt conversations
                    completions [list] the list of completions
                    scores      [list] of the yes/not answers
    """
    prompts         = []        # initialize the list of prompts
    completions     = []        # initialize the list of completions
    scores          = dict()    # initialize the yes/not replies
    img_names       = []        # initialize the list of image names
    pre             = ""        # optional preliminary dialog turn
    post            = ""        # optional post dialog turn, typically a query

    if not len( cnfg.news_ids ):
        # use all news in file if not specified otherwise
        cnfg.news_ids   = prmpt.list_news()

    if len( cnfg.dialogs ):
        pre         = prmpt.get_dialog( cnfg.dialogs[ 0 ] )
    if len( cnfg.dialogs ) > 1:
        post        = prmpt.get_dialog( cnfg.dialogs[ -1 ] )

    for n in cnfg.news_ids:
        if cnfg.VERBOSE:
            i_mode      = "with image" if with_img  else "without image"
            print( f"Processing news {n} {i_mode}" )

        pr, name        = prmpt.prompt_news( n, interface=cnfg.interface, pre=pre, post=post, with_img=with_img )

        # using OpenAI
        if cnfg.interface == "openai":
            completion  = cmplt.complete( pr )
            pr          = prmpt.prune_prompt( pr ) # remove the textual version of the image from the prompt
        # using HuggingFace
        else:
            image       = prmpt.image_pil( n ) if with_img else None
            completion  = cmplt.complete( pr, image=image )

        res             = check_reply( completion )
        scores[ n ]     = res
        prompts.append( pr )
        completions.append( completion )
        img_names.append( name )

    return prompts, completions, scores, img_names


def do_exec():
    """
    Execute the program in one of the available modality (with image, without, or both) and save the results.

    return:     True if execution is succesful
    """
    fstream         = open( exec_log, 'w', encoding="utf-8" )   # open the log file
    write_header( fstream )

    match cnfg.experiment:

        case "news_noimage":
            pr, compl, res, names   = ask_news( with_img=False )
            write_results( fstream, pr, compl, res, names )
            with open( exec_file, 'wb' ) as f:                  # save raw results in pickle file
                pickle.dump( ( res ), f )

        case "news_image":
            pr, compl, res, names   = ask_news( with_img=True )
            write_results( fstream, pr, compl, res, names )
            with open( exec_file, 'wb' ) as f:                  # save raw results in pickle file
                pickle.dump( ( res ), f )

        case "both":
            pr_img, com_img, res_img, n_i   = ask_news( with_img=True )
            pr_noi, com_noi, res_noi, n_n   = ask_news( with_img=False )
            prompts                         = pr_img + pr_noi
            completions                     = com_img + com_noi
            names                           = n_i + n_n
            results                         = { "with_img": res_img, "no_img": res_noi }
            write_results( fstream, prompts, completions, results, names )
            with open( exec_file, 'wb' ) as f:                  # save raw results in pickle file
                pickle.dump( ( results ), f )

        case _:
            print( f"WARNING: experiment '{cnfg.experiment}' not implemented" )
            return None

    fstream.close()
    return True



# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================

if __name__ == '__main__':
    if DEBUG:
        sys.exit()

    if DO_NOTHING:
        print( "Program instructed to DO_NOTHING" )

    else:
        init_cnfg()
        init_dirs()
        if cnfg.experiment is not None:
            archive()
            do_exec()
