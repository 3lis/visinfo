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
import  time
import  numpy       as np

import  load_cnfg                               # this module sets program parameters
import  prompt      as prmpt                    # this module composes the prompts
import  complete    as cmplt                    # this module performs LLM completions
import  save_res                                # this module saves results

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
exec_log                = 'log.txt'
exec_pkl                = 'res.pkl'
exec_csv                = 'res.csv'


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
    global exec_log, exec_pkl, exec_csv         # files

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
    exec_pkl        = os.path.join( exec_dir, exec_pkl )
    exec_csv        = os.path.join( exec_dir, exec_csv )


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
        cnfg.dialogs_pre    = ""                            # set a reasonable default
        cnfg.dialogs_post   = ""                            # set a reasonable default
        cnfg.news_ids       = []                            # set a reasonable default

    if not hasattr( cnfg, 'experiment' ):
        cnfg.experiment         = None                      # whether experiment uses images or not

    # overwrite command line arguments
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

    # pass global parameters to other modules
    cmplt.cnfg          = cnfg
    save_res.cnfg       = cnfg


def archive():
    """
    Save a copy of current python source and json data files in the execution folder
    """
    jfiles  = ( "dialogs.json",
                "news.json"
    )

    pfiles  = [ "main_exec.py",
                "prompt.py",
                "load_cnfg.py",
                "models.py",
                "save_res.py",
                "complete.py"
    ]

    if cnfg.CONFIG is not None:
        pfiles.append( cnfg.CONFIG + ".py" )

    for pfile in pfiles:
        shutil.copy( pfile, exec_src )
    for jfile in jfiles:
        jfile   = os.path.join( dir_json, jfile )
        shutil.copy( jfile, exec_data )



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

    if not len( cnfg.news_ids ):
        # use all news in file if not specified otherwise
        cnfg.news_ids   = prmpt.list_news()

    for n in cnfg.news_ids:
        if cnfg.VERBOSE:
            i_mode      = "img + txt" if with_img else "only txt"
            print( f"==========> Processing news {n} {i_mode} <==========" )

        pr, name        = prmpt.format_prompt(
                            n,
                            cnfg.interface,
                            pre         = cnfg.dialogs_pre,
                            post        = cnfg.dialogs_post,
                            with_img    = with_img,
                            source      = cnfg.info_source,
                            more        = cnfg.info_more,
        )

        # using OpenAI
        if cnfg.interface == "openai":
            completion  = cmplt.complete( pr )
            pr          = prmpt.prune_prompt( pr ) # remove the textual version of the image from the prompt
        # using HuggingFace
        else:
            image       = prmpt.image_pil( n ) if with_img else None
            completion  = cmplt.complete( pr, image=image )
            pr          = prmpt.prune_prompt( pr ) # remove the textual version of the image from the prompt

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

    match cnfg.experiment:
        case "news_noimage":
            pr, compl, res, names           = ask_news( with_img=False )

        case "news_image":
            pr, compl, res, names           = ask_news( with_img=True )

        case "both":
            pr_img, com_img, res_img, n_i   = ask_news( with_img=True )
            pr_noi, com_noi, res_noi, n_n   = ask_news( with_img=False )
            pr                              = pr_img + pr_noi
            compl                           = com_img + com_noi
            names                           = n_i + n_n
            res                             = { "with_img": res_img, "no_img": res_noi }

        case _:
            print( f"ERROR: experiment '{cnfg.experiment}' not implemented" )
            return None

    save_res.write_all( fstream, pr, compl, res, names, exec_csv, exec_pkl )
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
