"""
#####################################################################################################################

    Module to handle model completions

#####################################################################################################################
"""

import  os
import  sys
import  platform
from    PIL         import Image

key_file                = "../data/.key.txt"    # file with the current OpenAI API access key
hf_file                 = "../data/.hf.txt"     # file with the current huggingface access key

llava_next_res          = ( 672, 672 )          # image resolution accepted by LLaVA-NeXT or (336, 672) [HxW]
llava_next_n_max        = 50                    # maximum number of returns for LLaVA-NeXT (due to GPU memory)

client                  = None                  # the language model client object
cnfg                    = None                  # parameter obj assigned by main_exec.py


# ===================================================================================================================
#
#   - set_openai
#   - set_hf
#
#   - complete_openai
#   - complete_hf
#
#   - complete
#
# ===================================================================================================================

def set_openai():
    """
    Parse the OpenAI key and return the client
        NOTE: should be the first function to call before all others that use openai
    """
    from    openai          import OpenAI
    key             = open( key_file, 'r' ).read().rstrip()
    client          = OpenAI( api_key=key )
    return client


def set_hf():
    """
    Parse the hugginface key and return the client
        NOTE: should be the first function to call before all others that use hugginface models
        NOTE: the client has two items: the model and the prompt processor
    """
    global  torch
    from    transformers    import LlavaNextForConditionalGeneration, LlavaNextProcessor
    from    huggingface_hub import login
    import  torch

    key             = open( hf_file, 'r' ).read().rstrip()
    login( token=key )

    model           = LlavaNextForConditionalGeneration.from_pretrained(
            cnfg.model,
            torch_dtype=torch.float16,
            device_map="auto"
            )
    processor       = LlavaNextProcessor.from_pretrained( cnfg.model )
    client          = { "model": model, "processor": processor }
    return client


def complete_openai( prompt ):
    """
    Feed a prompt to an OpenAI model and get the list of completions returned.
    This function works for both models with "straight" completion, and models with chat completion.

    params:
        prompt      [str] or [list] the prompt for completion models,
                    or the messages for chat completion models

    return:         [list] with completions [str]
    """
    global client
    if cnfg.DEBUG:  return [ "test_only" ]

    if client is None:              # check if openai has already a client, otherwise set it
        client  = set_openai()
    user    = os.getlogin() + '@' + platform.node()

    if cnfg.mode == "complete":
        assert isinstance( prompt, str ), "ERROR: for complete models the prompt should be a string"
        res     = client.completions.create(
            model                   = cnfg.model,
            prompt                  = prompt,
            max_tokens              = cnfg.max_tokens,
            n                       = cnfg.n_returns,
            top_p                   = cnfg.top_p,
            temperature             = cnfg.temperature,
            stop                    = None,
            user                    = user
        )
        return [ t.text for t in res.choices ]

    if cnfg.mode == "chat":
        assert isinstance( prompt, list ), "ERROR: for chat complete models the prompt should be a list"
        # try to avoid rate limits by catching openai.error.RateLimitError exception and just sleeping for a while
        # and then repeating the same completion.
        # NOTE: for gpt-4o stop=None raises Error code: 400! do not use it
        res     = client.chat.completions.create(
        model                   = cnfg.model,
        messages                = prompt,
        max_tokens              = cnfg.max_tokens,
        n                       = cnfg.n_returns,
        top_p                   = cnfg.top_p,
        temperature             = cnfg.temperature,
        user                    = user)
        return [ t.message.content for t in res.choices ]

    return None


def complete_hf( prompt, image ):
    """
    Feed a prompt to a HuggingFace model and get the list of completions returned.
    This function doesn't work for models with "straight" completion, only with chat completion.

        NOTE: currently there is a bug in llava-next when doing inference without an image:
        https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/discussions/36
        As a temporary workaround, if no image is requested, a blank image of the same size is loaded.

    params:
        prompt      [str] or [list] the prompt for completion models,
                    or the messages for chat completion models
        image       [PIL.JpegImagePlugin.JpegImageFile] or None in case of no image

    return:         [list] with completions [str]
    """
    global client
    if cnfg.DEBUG:  return [ "test_only" ]

    if client is None:              # check if hf has already a client, otherwise set it
        client  = set_hf()

    if cnfg.mode == "complete":
        print( "WARNING: complete_hf() not yet implemented for complete-mode models" )
        sys.exit()

    assert isinstance( prompt, list ), "ERROR: for chat complete models the prompt should be a list"

    model       = client[ "model" ]
    processor   = client[ "processor" ]
    text        = processor.apply_chat_template( prompt, add_generation_prompt=True )

    # dummy image as workaround for llava-next bug
    if image is None:
        image   = Image.new( mode='L', size=llava_next_res, color="black" )
    else:
        image   = image.resize( llava_next_res )

    inputs      = processor(
            images          = image,
            text            = text,
            return_tensors  = "pt"
        ).to( model.device, torch.float16 )

    # to suppress annoying print
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # NOTE: for the list of arguments see the general documentation for Generation
    # https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/text_generation
    out         = model.generate(
            **inputs,
            max_new_tokens          = cnfg.max_tokens,
            do_sample               = True,                     # NOTE: the default is greedy!
            num_return_sequences    = cnfg.n_returns,
            top_p                   = cnfg.top_p,
            temperature             = cnfg.temperature,
            # pad_token_id            = tokenizer.eos_token_id    # to suppress annoying print
        )
    res         = processor.batch_decode( out, skip_special_tokens=True )
    completions = [ r.split( "[/INST]" )[ -1 ].strip() for r in res ]

    return completions


def complete( prompt, image=None ):
    """
    Feed a prompt to any model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion models,
                    or the messages for chat completion models
        image       [PIL.JpegImagePlugin.JpegImageFile] or None, for OpenAI the image is embedded in the propmt

    return:         [list] with completions [str]
    """
    match cnfg.interface:

        case 'openai':
            return complete_openai( prompt )

        case 'hf':
            if cnfg.n_returns <= llava_next_n_max:
                return complete_hf( prompt, image=image )
            # to avoid problems of GPU memory, do separate reps
            reps            = cnfg.n_returns // llava_next_n_max
            reminder        = cnfg.n_returns % llava_next_n_max
            saved           = cnfg.n_returns                    # make a copy of the original number of results
            completions     = []
            cnfg.n_returns  = llava_next_n_max
            for i in range( reps ):
                completions += ( complete_hf( prompt, image=image ) )
            if reminder:
                cnfg.n_returns  = reminder
                completions += ( complete_hf( prompt, image=image ) )
            cnfg.n_returns  = saved                             # restore the original number of results
            return completions

        case _:
            print( f"WARNING: model interface '{cnfg.interface}' not supported" )
            return None
