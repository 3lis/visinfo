"""
#####################################################################################################################

    Module to read and set configuration parameters

#####################################################################################################################
"""


import  os
from    argparse        import ArgumentParser


class Config( object ):
    """
    Object containing all parameters accepted by the software.
    Several parameters can be given in the configuration file as well as with command line flags.

    Command line flags:
    CONFIG                  [str] name of configuration file (without path nor extension) (DEFAULT=None)
    DEBUG                   [str] debug mode, for generic debugging in selected parts of the software
    MAXTOKENS               [int] maximum number of tokens (DEFAULT=None)
    MODEL                   [int] index in the list of possible models (DEFAULT=0)
    NRETURNS                [int] number of return sequences (DEFAULT=None)
    VERBOSE                 [bool] write additional information

    Configuration file parameters:
    demographics            [dic] demographic data or None
    detail                  [str] detail parameter for OpenAI image handling: "high", "low", "auto"
    dialogs_pre             [list or str] dialog ids to instert before the news
    dialogs_post            [list or str] dialog ids to instert after the news
    f_dialog                [str] filename of json file with dialogs
    f_news                  [str] filename of json file with the news
    info_source             [bool] add info about the source of the news
    info_more               [bool] add more available info about the news, like number of share/followers
    model_id                [int] index in the list of possible models (overwritten by MODEL)
    max_tokens              [int] maximum number of tokens (overwritten by MAXTOKENS)
    n_returns               [int] number of return sequences (overwritten by NRETURNS)
    news_ids                [list] ids of news to process
    repetition_penalty      [float] penality for text repetitions in completion
    top_p                   [int] probability mass of tokens generated in completion (default=1)
    temperature             [float] sampling temperature during completion (default=1.0)

    """

    def load_from_line( self, line_kwargs ):
        """
        Load parameters from command line arguments

        params:
            line_kwargs     [dict] parameteres read from arguments passed in command line
        """
        for key, value in line_kwargs.items():
            setattr( self, key, value )


    def load_from_file( self, file_kwargs ):
        """
        Load parameters from a python file.
        Check the correctness of parameteres, set defaults.

        params:
            file_kwargs     [dict] parameteres coming from a python module (file)
        """
        for key, value in file_kwargs.items():
            setattr( self, key, value )

        if not hasattr( self, 'news_ids' ):
            self.news_ids           = []
        if not hasattr( self, 'dialogs_pre' ):
            self.dialogs_pre        = ""
        if not hasattr( self, 'dialogs_post' ):
            self.dialogs_post       = ""
        if not hasattr( self, 'model_id' ):
            self.model_id           = 0
        if not hasattr( self, 'n_returns' ):
            self.n_returns          = 1
        if not hasattr( self, 'max_tokens' ):
            self.max_tokens         = 20
        if not hasattr( self, 'top_p' ):
            self.top_p              = 1
        if not hasattr( self, 'temperature' ):
            self.temperature        = 0.3
        if not hasattr( self, 'info_source' ):
            self.info_source        = False
        if not hasattr( self, 'info_more' ):
            self.info_more          = False
        if not hasattr( self, 'repetition_penalty' ):
            self.repetition_penalty = 1.1
        if not hasattr(self, 'demographics'):
            self.demographics       = None      # Default is not including demographics


    def __str__( self ):
        """
        Visualize the list of all parameters
        """
        s   = ''
        d   = self.__dict__

        for k in d:
            if isinstance( d[ k ], dict ):
                s   += "{}:\n".format( k )
                for j in d[ k ]:
                    s   += "{:5}{:<30}{}\n".format( '', j, d[ k ][ j ] )
            else:
                s   += "{:<35}{}\n".format( k, d[ k ] )

        return s


# ===================================================================================================================


def read_args():
    """
    Parse the command-line arguments defined by flags

    return:         [dict] key = name of parameter, value = value of parameter
    """
    parser      = ArgumentParser()

    parser.add_argument(
            '-c',
            '--config',
            action          = 'store',
            dest            = 'CONFIG',
            type            = str,
            default         = None,
            help            = "Name of configuration file (without path nor extension)"
    )
    parser.add_argument(
            '-D',
            '--debug',
            action          = 'store_true',
            dest            = 'DEBUG',
            help            = "debug mode: print prompts only, do not call LLMs"
    )
    parser.add_argument(
            '-m',
            '--model',
            action          = 'store',
            dest            = 'MODEL',
            type            = int,
            default         = None,
            help            = "index in the list of possible models (default=0) (-1 to print all)",
    )
    parser.add_argument(
            '-M',
            '--maxreturns',
            action          = 'store',
            dest            = 'MAXTOKENS',
            type            = int,
            default         = None,
            help            = "maximum number of tokens (default=500)",
    )
    parser.add_argument(
            '-n',
            '--nreturns',
            action          = 'store',
            dest            = 'NRETURNS',
            type            = int,
            default         = None,
            help            = "number of return sequences (default=1)",
    )
    parser.add_argument(
            '-v',
            '--verbose',
            action          = 'store_true',
            dest            = 'VERBOSE',
            help            = "write additional information"
    )

    return vars( parser.parse_args() )
