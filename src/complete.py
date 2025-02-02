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
qwen2_vl_n_max          = 2                     # maximum number of returns for Qwen2-VL-2B (due to GPU memory)
qwen2_vl_min_res        = 224*224               # minimum image resolution for Qwen2-VL-2B
qwen2_vl_max_res        = 672*672               # maximum image resolution for Qwen2-VL-2B

client                  = None                  # the language model client object
cnfg                    = None                  # parameter obj assigned by main_exec.py


# ===================================================================================================================
#
#   - set_hf_llava_next
#   - set_hf_chameleon
#   - set_hf_qwen
#   - set_hf
#   - set_openai
#
# ===================================================================================================================

def set_hf_llava_next():
    """
    Return the LlavaNext client
    """
    global  torch
    from    transformers    import LlavaNextForConditionalGeneration, LlavaNextProcessor
    import  torch

    model           = LlavaNextForConditionalGeneration.from_pretrained(
            cnfg.model,
            torch_dtype=torch.float16,
            device_map="auto"
            )
    processor       = LlavaNextProcessor.from_pretrained( cnfg.model )
    client          = { "model": model, "processor": processor }
    return client


def set_hf_chameleon():
    """
    Return the Chameleon client
    """
    global  torch
    from    transformers    import ChameleonForConditionalGeneration, ChameleonProcessor
    import  torch

    model           = ChameleonForConditionalGeneration.from_pretrained(
            cnfg.model,
            torch_dtype         = torch.float16,
            repetition_penalty  = cnfg.repetition_penalty,
            device_map          = "auto"
            )
    processor       = ChameleonProcessor.from_pretrained( cnfg.model )
    client          = { "model": model, "processor": processor }
    return client


def set_hf_qwen():
    """
    Return the Qwen client
    """
    global  torch
    global  process_vision_info
    from    transformers    import Qwen2VLForConditionalGeneration, AutoProcessor
    from    qwen_vl_utils   import process_vision_info
    import  torch

    model           = Qwen2VLForConditionalGeneration.from_pretrained(
            cnfg.model,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",    # should install FlashAttention-2 and see if works
            device_map="auto"
            )
    processor       = AutoProcessor.from_pretrained(
            cnfg.model,
            min_pixels  = qwen2_vl_min_res,
            max_pixels  = qwen2_vl_max_res
            )
    client          = { "model": model, "processor": processor }
    return client


def set_hf():
    """
    Parse the hugginface key and return the client
        NOTE: should be the first function to call before all others that use hugginface models
        NOTE: the client has two items: the model and the prompt processor
    """
    from    huggingface_hub import login

    key             = open( hf_file, 'r' ).read().rstrip()
    login( token=key )

    if "llava-v1.6" in cnfg.model:
        return set_hf_llava_next()
    if "chameleon" in cnfg.model:
        return set_hf_chameleon()
    if "Qwen" in cnfg.model:
        return set_hf_qwen()

    return None


def set_openai():
    """
    Parse the OpenAI key and return the client
        NOTE: should be the first function to call before all others that use openai
    """
    from    openai          import OpenAI
    key             = open( key_file, 'r' ).read().rstrip()
    client          = OpenAI( api_key=key )
    return client


# ===================================================================================================================
#
#   - complete_openai
#   - complete_llava
#   - complete_hf
#   - complete_hf
#   - complete_hf
#   - complete_hf
#
#   - do_complete
#
# ===================================================================================================================

def complete_openai( prompt ):
    """
    Feed a prompt to an OpenAI model and get the list of completions returned.
    This function works for both completion-mode models and chat-mode models.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models

    return:         [list] with completions [str]
    """
    global client
#   if cnfg.DEBUG:  return [ "test_only" ]

    if client is None:              # check if openai has already a client, otherwise set it
        client  = set_openai()
    user    = os.getlogin() + '@' + platform.node()

    if cnfg.mode == "cmpl":
        assert isinstance( prompt, str ), "ERROR: for completion-mode models, the prompt should be a string"
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
        assert isinstance( prompt, list ), "ERROR: for chat-mode models, the prompt should be a list"
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
            user                    = user
        )
        return [ t.message.content for t in res.choices ]

    return None


def complete_llava( model, processor, prompt, image ):
    """
    Feed a prompt to a Llava model and get the list of completions returned.

        NOTE: currently there is a bug in llava-next when doing inference without an image:
        https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/discussions/36
        As a temporary workaround, if no image is requested, a blank image of the same size is loaded.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models
        image       [PIL.JpegImagePlugin.JpegImageFile] or None in case of no image

    return:         [list] with completions [str]
    """
    text        = processor.apply_chat_template( prompt, add_generation_prompt=True )

    # dummy image as workaround for llava-next bug (see comment above)
    if image is None:
        image   = Image.new( mode='L', size=llava_next_res, color="black" )
    else:
        image   = image.resize( llava_next_res )

    inputs      = processor(
            images          = image,
            text            = text,
            return_tensors  = "pt"
    ).to( model.device, torch.float16 )

    out         = model.generate(
            **inputs,
            max_new_tokens          = cnfg.max_tokens,
            do_sample               = True,                     # NOTE: the default is greedy!
            num_return_sequences    = cnfg.n_returns,
            top_p                   = cnfg.top_p,
            temperature             = cnfg.temperature,
    )
    res         = processor.batch_decode( out, skip_special_tokens=True )

    # NOTE that the prompt is included in the completion, there is no parameter like return_full_text in pipeline
    # that can avoid this issue in model.generate. Therefore, here are workarounds that are model dependent.
    # CHECK when new models are added
    end_input   = "[/INST]"
    completions = [ r.split( end_input )[ -1 ].strip() for r in res ]

    return completions


def complete_chameleon( model, processor, prompt, image ):
    """
    Feed a prompt to a Chameleon model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models
        image       [PIL.JpegImagePlugin.JpegImageFile] or None in case of no image
        model       [transformers.models...] client model
        processor   transformers.models...] client input processor

    return:         [list] with completions [str]
    """
    if image is None:
        text        = prompt
    else:
        text        = prompt + "<image>"
        image       = image.resize( llava_next_res )

    inputs      = processor(
            images          = image,
            text            = text,
            return_tensors  = "pt"
    ).to( model.device, torch.float16 )

    out         = model.generate(
            **inputs,
            max_new_tokens          = cnfg.max_tokens,
            do_sample               = True,                     # NOTE: the default is greedy!
            num_return_sequences    = cnfg.n_returns,
            top_p                   = cnfg.top_p,
            temperature             = cnfg.temperature,
    )
    res         = processor.batch_decode( out, skip_special_tokens=True )

    # NOTE that the prompt is included in the completion, there is no parameter like return_full_text in pipeline
    # that can avoid this issue in model.generate. Therefore, here are workarounds that are model dependent.
    # CHECK when new models are added
    completions = [ r.replace( prompt, "" ) for r in res ]

    return completions


def complete_qwen( model, processor, prompt ):
    """
    Feed a prompt to a Qwen model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models

    return:         [list] with completions [str]
    """
    text        = processor.apply_chat_template(
                    prompt,
                    tokenize                = False,
                    add_generation_prompt   = True
    )
    imgs, video = process_vision_info( prompt )
    inputs      = processor(
            text            = text,
            images          = imgs,
            videos          = video,
            padding         = True,
            return_tensors  = "pt"
    ).to( model.device, torch.float16 )

    out         = model.generate(
            **inputs,
            max_new_tokens          = cnfg.max_tokens,
            do_sample               = True,                     # NOTE: the default is greedy!
            num_return_sequences    = cnfg.n_returns,
            top_p                   = cnfg.top_p,
            temperature             = cnfg.temperature,
    )
    res         = processor.batch_decode( out, skip_special_tokens=True )

    # NOTE that the prompt is included in the completion, there is no parameter like return_full_text in pipeline
    # that can avoid this issue in model.generate. Therefore, here are workarounds that are model dependent.
    # CHECK when new models are added
    end_input   = "\nassistant\n"
    completions = [ r.split( end_input )[ -1 ].strip() for r in res ]

    return completions


def complete_hf( prompt, image ):
    """
    Feed a prompt to a HuggingFace model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models
        image       [PIL.JpegImagePlugin.JpegImageFile] or None in case of no image

    return:         [list] with completions [str]
    """
    global client
#   if cnfg.DEBUG:  return [ "test_only" ]

    if client is None:              # check if hf has already a client, otherwise set it
        client  = set_hf()

    model       = client[ "model" ]
    processor   = client[ "processor" ]

    if "llava-v1.6" in cnfg.model:
        return complete_llava( model, processor, prompt, image )
    if "chameleon" in cnfg.model:
        return complete_chameleon( model, processor, prompt, image )
    if "Qwen" in cnfg.model:
        return complete_qwen( model, processor, prompt )

    print( f"WARNING: model '{cnfg.model}' not currently supported.")
    return None


def do_complete( prompt, image=None ):
    """
    Feed a prompt to any model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion models,
                    or the messages for chat completion models
        image       [PIL.JpegImagePlugin.JpegImageFile] or None, for OpenAI and Qwen
                    the image is embedded in the propmt

    return:         [list] with completions [str]
    """
    match cnfg.interface:

        case 'openai':
            return complete_openai( prompt )

        case 'hf':
            if "Qwen" in cnfg.model:
                n_max   = qwen2_vl_n_max
            else:
                n_max   = llava_next_n_max
            if cnfg.n_returns <= n_max:
                return complete_hf( prompt, image=image )
            # to avoid problems of GPU memory, do separate reps
            reps            = cnfg.n_returns // n_max
            reminder        = cnfg.n_returns % n_max
            saved           = cnfg.n_returns                    # make a copy of the original number of results
            completions     = []
            cnfg.n_returns  = n_max
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
