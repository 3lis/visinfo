kwargs      = {
    'model_id':         8,                  # chameleon-7b
    'experiment':       "both",             # news_image, news_noimage, both

    # 'dialogs_pre':      [ "intro_profile", "profile_moderate", "context_strict" ],
    'dialogs_pre':      [ "intro_profile", "profile_conspirator", "context_strict" ],
    # 'dialogs_pre':      [ "intro_profile", "profile_rational", "context_strict" ],

    'dialogs_post':     [ "ask_img", "ask_share_strict"],
    # 'news_ids':         [ 'f001', 'f002', 'f003', 'f004' ],

    'max_tokens':       500,
    'n_returns':        10,
    # 'n_returns':        4,
    'top_p':            1.0,
    'temperature':      0.3,
    'info_source':      False,
    'info_more':        False,
}
