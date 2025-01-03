kwargs      = {
    'model_id':         -1,                 # llava-next
    'experiment':       "both",             # news_image, news_noimage, both
    'dialogs_pre':      [ "intro_profile", "profile_1", "intro_context" ],
    'dialogs_post':     [ "ask_img", "ask_share"],
    # 'news_ids':         [ 'f001', 'f002' ],
    'max_tokens':       300,
    'n_returns':        5,
    'top_p':            1.0,
    'temperature':      0.3,
    'info_source':      False,
    'info_more':        False,
}
