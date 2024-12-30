"""
#####################################################################################################################

    Module to compose prompts

#####################################################################################################################
"""


import  os
import  sys
import  string
import  base64
import  json
from    PIL         import Image


dir_json                = "../data"                 # directory with all input data
dir_imgs                = "../imgs"                 # directory with news images
f_dialog                = "dialogs.json"            # filename of the preliminary dialogs
f_news                  = "news.json"               # filename of the news tests

img_width               = 720                       # typical width of news images (not the same for all)
img_height              = 290                       # typical height of news images (not the same for all)
img_spacing             = 40                        # spacing between top and bottom images in a composition
img_offset              = 10                        # offset at the border of news images
detail                  = "high"                    # detail parameter for OpenAI image handling, overwritten by cnfg


# ===================================================================================================================
#
#   Utilities to read data
#   - list_news
#   - list_dialogs
#   - image_pil
#   - image_b64
#   - get_dialog
#
# ===================================================================================================================

def list_news():
    """
    Return the list with all ID of the news found in the JSON dataset

    return:     [list] with news IDs
    """
    fname   = os.path.join( dir_json, f_news )
    with open( fname, 'r' ) as f:
        data    = json.load( f )
    ids     = [ int( d[ 'id' ] ) for d in data ]

    return ids


def list_dialogs():
    """
    Return the list with all ID of the dialogs found in the JSON dataset

    return:     [list] with dialog IDs
    """
    fname   = os.path.join( dir_json, f_dialog )
    with open( fname, 'r' ) as f:
        data    = json.load( f )
    ids     = [ d[ 'id' ] for d in data ]

    return ids


def image_pil( i ):
    """
    Return an image as PIL object, as requested in LlaVa prompts

    params:
        i       [int] id of the news linked to the image

    return:     [PIL.JpegImagePlugin.JpegImageFile]
    """
    ids     = list_news()
    try:
        idx     = ids.index( i )
    except Exception as e:
        print( f"ERROR: non existing news with ID={i} in image_pil()" )
        raise e

    img_name    = data[ idx ][ "image" ]
    fname       = os.path.join( dir_imgs, img_name )
    img         = Image.open( fname )
    return img


def image_b64( fname ):
    """
    Return an image as b64encoded string, as requested in OpenAi prompts

    params:
        fname   [str] name of the file, without path

    return:     [bytes] the b64encoded image
    """
    fname       = os.path.join( dir_imgs, fname )
    with open( fname, 'rb' ) as f:
        img_str     = base64.b64encode( f.read() ).decode( "utf-8" )

    return img_str


def get_dialog( i ):
    """
    Return a single dialog from the JSON dataset

    params:
        i       [str] ID of the dialog

    return:     [str] the dialog content
    """
    ids         = list_dialogs()
    try:
        idx     = ids.index( i )
    except Exception as e:
        print( f"ERROR: non existing dialog '{i}' in get_dialog()" )
        raise e
    text    = data[ idx ][ "content" ]
    return text



# ===================================================================================================================
#
#   Functions composing prompts
#   - prune_prompt
#   - prompt_news
#
# ===================================================================================================================

def prune_prompt( prompt ):
    """
    Prune the b64-encoded images from the prompt.
    This is used to save the textual part only in the log.

    params:
        prompt  [list] the prompt

    return:     [list] the pruned prompt
    """
    pruned      = []

    for p in prompt:
        if isinstance( p[ "content" ], str ):
            pruned.append( p )
        else:
            role    = p[ "role" ]
            if "text" in p[ "content" ][ 0 ]:
                text    = p[ "content" ][ 0 ][ "text" ]
                pruned.append( { "role": role, "content": text } )

    return pruned


def prompt_news( news_id, interface="openai", pre="", post="", with_img=True ):
    """
    Compose the prompt to process one news.
    For OpenAI interface, the image is passed within the prompt.
    For HF interface, the image is passed separately in complete.py

    params:
        news        [int] the news item to test
        interface   [str] "openai" or "hf"
        pre         [str] optional text before the news content
        post        [str] optional text after the news content
        with_img    [bool] combine the news with the image

    return:         [list] the prompt
                    [str] image name or "" if not with_img
    """
    fname   = os.path.join( dir_json, f_news )
    with open( fname, 'r' ) as f:
        data    = json.load( f )
    ids     = [ int( d[ 'id' ] ) for d in data ]

    try:
        idx     = ids.index( news_id )
    except Exception as e:
        print( f"ERROR: non existing news with ID {news_id} in prompt_news()" )
        raise e

    news                = data[ idx ]
    title               = news[ "title" ]
    text                = news[ "content" ]
    fimage              = news[ "image" ]

    full_text           = f"{pre}\n{title} - {text}\n{post}"

    if interface == "openai":
        # OpenAI with image included as string in the prompt
        if with_img:
            image               = image_b64( fimage )
            img_content         = {
                    "type":         "image_url",
                    "image_url" :   {
                        "url":      f"data:image/jpeg;base64,{image}",
                        "detail":   detail
                    }
                }
            prompt      = [ {
                "role":     "user",
                "content":  [
                    { "type": "text", "text": full_text },
                    img_content
                ]
            } ]
            return prompt, fimage

        # OpenAI without image
        else:
            prompt      = [ {
                "role":     "user",
                "content":  [ { "type": "text", "text": full_text } ]
            } ]
            return prompt, ''

    # HuggingFace with or without image
    # NOTE currently huggingface has a bug that does not allow inference without an image (see complete.py)
    img_content         = { "type": "image" }
    prompt      = [ {
        "role":     "user",
        "content":  [
            { "type": "text", "text": full_text },
            { "type": "image" }
        ]
    } ]
    return prompt, fimage
    
