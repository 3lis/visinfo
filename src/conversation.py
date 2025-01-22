"""
#####################################################################################################################

    Module to handle conversation dynamics

#####################################################################################################################
"""

import  numpy           as np

import  prompt          as prmpt                # this module composes the prompts
import  complete        as cmplt                # this module performs LLM completions

cnfg                    = None                  # parameter obj assigned by main_exec.py

# ===================================================================================================================
#
#   - check_reply
#   - ask_news
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

        if "<yes>" in c:
            res[ i ]    = True
        elif "<no>" in c:
            res[ i ]    = False
        else:
            print( f"WARNING: unclear reply to YES/NO question. Considering <NO> as answer." )
            res[ i ]    = False

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

        interface       = "qwen"    if "Qwen" in cnfg.model     else cnfg.interface
        
        pr, name        = prmpt.format_prompt(
                            n,
                            interface,
                            mode        = cnfg.mode,
                            pre         = cnfg.dialogs_pre,
                            post        = cnfg.dialogs_post,
                            with_img    = with_img,
                            source      = cnfg.info_source,
                            more        = cnfg.info_more,
        )

        # using OpenAI
        if cnfg.interface == "openai":
            completion  = cmplt.do_complete( pr )
            pr          = prmpt.prune_prompt( pr ) # remove the textual version of the image from the prompt
        # using HuggingFace
        else:
            image       = prmpt.image_pil( n ) if with_img else None
            completion  = cmplt.do_complete( pr, image=image )

        res             = check_reply( completion )
        scores[ n ]     = res
        prompts.append( pr )
        completions.append( completion )
        img_names.append( name )

    return prompts, completions, scores, img_names
