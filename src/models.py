"""
#####################################################################################################################

    list of models in use and their properties

#####################################################################################################################
"""

models                  = (                     # available models (first one is the default)
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-vision-preview",
        "gpt-4o-2024-05-13",
        "gpt-4o",
        "gpt-4o-mini",
        "llava-hf/llava-v1.6-mistral-7b-hf",
)
models_endpoint         = {                     # which endpoint should be used for a model
        "gpt-3.5-turbo-instruct"            : "complete",
        "gpt-3.5-turbo"                     : "chat",
        "gpt-4"                             : "chat",
        "gpt-4-vision-preview"              : "chat",
        "gpt-4o-2024-05-13"                 : "chat",
        "gpt-4o"                            : "chat",
        "gpt-4o-mini"                       : "chat",
        "llava-hf/llava-v1.6-mistral-7b-hf" : "chat",
}
models_interface        = {                     # which interface should be used for a model
        "gpt-3.5-turbo-instruct"            : "openai",
        "gpt-3.5-turbo"                     : "openai",
        "gpt-4"                             : "openai",
        "gpt-4-vision-preview"              : "openai",
        "gpt-4o-2024-05-13"                 : "openai",
        "gpt-4o"                            : "openai",
        "gpt-4o-mini"                       : "openai",
        "llava-hf/llava-v1.6-mistral-7b-hf" : "hf",
}
