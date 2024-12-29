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


data_dir                = "../data"                 # directory with all input data
img_dir                 = "../imgs"                 # directory with news images
f_dialog                = "dialogs.json"            # filename of the preliminary dialogs
f_tasks                 = "tasks.json"              # filename of the Kosinski ToM tasks
f_news                  = "news.json"               # filename of the news tests

img_width               = 720                       # typical width of news images (not the same for all)
img_height              = 290                       # typical height of news images (not the same for all)
img_spacing             = 40                        # spacing between top and bottom images in a composition
img_offset              = 10                        # offset at the border of news images
detail                  = "high"                    # detail parameter for OpenAI image handling, overwritten by cnfg


def n_news():
    """
    return the list with all ID of the news found in the dataset

    return:     [list] with news IDs
    """
    fname       = os.path.join( data_dir, f_news )
    with open( fname, 'r' ) as f: data = json.load( f )
    n_list      = [ d[ 'id' ] for d in data ]

    return n_list


def jpg_image( i ):
    """
    return one news image as PIL jpeg format object
    as requested in Lava prompt

    i           [int] number of the news

    return:     [tuple] ( [PIL.JpegImagePlugin.JpegImageFile], image_name )
    """
    fname       = os.path.join( data_dir, f_news )
    with open( fname, 'r' ) as f: data = json.load( f )
    ids         = [ int( d[ 'id' ] ) for d in data ]
    try:
        idx     = ids.index( i )
    except Exception as e:
        print( f"non existing news with ID {i} in jpg_image()" )
        raise e
    img_name    = data[ idx ][ "image" ]
    fname       = os.path.join( img_dir, img_name )
    img         = Image.open( fname )

    return img, img_name


def one_image( fname ):
    """
    convert one news image into b64encoded string, as requested in the
    message for the OpenAi model

    fname       [str] name of the file, without path and extension

    return:     [bytes] the b64encoded image
    """
    fname       = os.path.join( img_dir, fname )
    with open( fname, 'rb' ) as f:
        img_str     = base64.b64encode( f.read() ).decode( "utf-8" )

    return img_str


def insert_vars( s, var ):
    """
    insert the values of independent variables if present in the given string
    placeholders for variable should follow the standard syntax
        {var_name}
    if var_name is found as key in the dictionary var the string is filled with the value

    s:          [str] the input string
    var:        [dic] with variable names as keys and their values with [str] format

    return:     [list]
    """
    # need to gather the names of variables (if any), not that straightforward,
    # after quite a lot of experiments, this instruction does the trick...
    var_list    = [ v for _, v, _, _ in string.Formatter().parse( s ) if v is not None ]

    if not len( var_list ):             # if there is no slot for variables, return the unmodified string
        return s

    for v in var_list:                  # if a variable is not in the given dictionary raise an error
        assert v in var.keys(), f"error in insert_vars: missing variable {v}"

    return s.format( **var )            # fill the slots with the value of the variables


def dialog_prompt( i ):
    """
    get a portion of prompt from the dialog dataset

    i:          [int] "id" field in the dialogs

    return:     [str] the prompt partial content
    """

    prompt      = []
    fname       = os.path.join( data_dir, f_dialog )
    with open( fname, 'r' ) as f: data = json.load( f )
    ids         = [ d[ 'id' ] for d in data ]
    try:
        idx     = ids.index( i )
    except Exception as e:
        print( f"non existing dialog {i} in dialog_prompt()" )
        raise e
    text    = data[ idx ][ "content" ]

    return text


def prune_news_prompt( prompt ):
    """
    prune the large base64 encoded images from the prompt, to save the relevant part only

    prompt:     [list] the propmt

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


def news_prompt( news_id, interface="openai", pre="", post="", with_img=True ):
    """
    compose the prompt for processing one news
    taking into account the language model interface

    news        [int] the news item to test
    interface   [str] "openai" or "hf"
    with_img    [bool] combine the news with the image

    return:     [tuple] ( [list] the prompt, [str] image name, empty string if not with_img )
    """

    fname           = os.path.join( data_dir, f_news )
    with open( fname, 'r' ) as f: data = json.load( f )
    ids         = [ int( d[ 'id' ] ) for d in data ]

    try:
        idx     = ids.index( news_id )
    except Exception as e:
        print( f"non existing news with ID {news_id} in news_prompt()" )
        raise e
    news                = data[ idx ]
    text                = news[ "content" ]
    fimage              = news[ "image" ]
    text                = pre + text + post

    if with_img:                    # include directive for processing the image
        if interface == "openai":   # OpenAI includes the image as string in the prompt
            image               = one_image( fimage )
            img_content         = {
                    "type": "image_url",
                    "image_url" : {
                        "url":      f"data:image/jpeg;base64,{image}",
                        "detail":   detail
                    }
                }
        else:                       # huggingface do not include the image in the prompt
            img_content         = { "type": "image" }
        prompt      = [ {
            "role":     "user",
            "content":  [
            {
                "type": "text",
                "text": text
            },
            img_content
            ]
        } ]
        return prompt, fimage

    prompt      = [ {           # no image processing directive
        "role":     "user",
        "content":  [
        {
            "type": "text",
            "text": text
        }
        ]
    } ]
    return prompt, ''
