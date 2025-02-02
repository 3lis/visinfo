"""
#####################################################################################################################

    Module to handle conversation dynamics

#####################################################################################################################
"""

import  sys
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

    return:         three [np.array] of booleans for yes/no/unk replies
    """
    no_patterns = [
        "<decision>no",
        "would recommend not reposting",
        "would recommend not sharing",
        "would be advisable to not repost",
        "would be advisable to not share",
        "i would not repost",
        "i would not post",
        "i would not share",
        "i would not recommend reposting",
        "i would not recommend sharing",
        "i should not repost",
        "i should not share",
        "i would not feel comfortable sharing",
        "i do not want to repost",
        "i do not want to share",
        "i would not want to repost",
        "i would not want to share",
    ]
    yes_patterns    = [
        "<decision>yes",
        "i want to repost",
        "i want to share",
        "i would recommend doing so",
        "i would decide to repost",
        "i would decide to share",
        "i would want to repost",
        "i would want to share",
        "i would feel inclined to repost",
        "i would feel inclined to share",
        "i would likely share",
        "i would like to repost",
        "i would like to share",
        "i might consider sharing it",
        "it would be reasonable to share",
    ]

    values      = 'yes', 'no', 'unk'
    nc          = len( completion )
    res         = dict()
    for v in values:
        res[ v ]    = np.full( nc, False )

    for i, c in enumerate( completion ):
        c       = c.lower()
        found   = False

        # first, check if reply contains <yes>/<no>
        if "<yes>" in c:
            res[ "yes" ][ i ]   = True
            found               = True
        elif "<no>" in c:
            res[ "no" ][ i ]    = True
            found               = True

        # second, check if reply starts with yes/no
        elif c[ :3 ] == "yes":
            res[ "yes" ][ i ]   = True
            found               = True
        elif c[ :2 ] == "no":
            res[ "no" ][ i ]    = True
            found               = True

        # third, check if reply contains positive/negative patterns
        else:
            found_yes       = any( p in c for p in yes_patterns )
            found_no        = any( p in c for p in no_patterns )

            if found_yes and found_no:
                print( f"WARNING: contradicting reply to YES/NO question. Considering <UNK> as answer." )
                res[ "unk" ][ i ]       = True
                if cnfg.DEBUG:
                    sys.exit()

            if found_yes:
                res[ "yes" ][ i ]       = True
                found                   = True
            elif found_no:
                res[ "no" ][ i ]        = True
                found                   = True

        # if not found:
        #     for p in yes_patterns:
        #         if p in c:
        #             res[ "yes" ][ i ]   = True
        #             found               = True
        #             break
        #         else:
        #             if cnfg.DEBUG:
        #                 print( f"not found: {p}\n in: {c}" )
        #
        # if not found:
        #     for p in no_patterns:
        #         if p in c:
        #             res[ "no" ][ i ]    = True
        #             found               = True
        #             break
        #         else:
        #             if cnfg.DEBUG:
        #                 print( f"not found: {p}\n in: {c}" )

        # otherwise, default to unknown
        if not found:
            print( f"WARNING: unclear reply to YES/NO question. Considering <UNK> as answer." )
            res[ "unk" ][ i ]    = True
            if cnfg.DEBUG:
                sys.exit()

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
