import os
import configparser
import logging as log

class CConfiguration:
    """
    Class containing configuration of the application.

    Parameters
        p_filename      (str)   : The path of the file to be read to construct the configuration instance
        -------------------------
        p_app_env       (str)   : The type of environment in which the application runs
        p_app_debug     (str)   : The configuration (Debug/Release) of the running application. Value should be `True` or `False` 
        p_fs_workspace  (str)   : The path to the workspace of the running application
        p_fs_weightsdir (str)   : The path to the directory containing model's weights
        p_fs_datadir    (str)   : The path to the directory containing data
        p_fs_resultsdir (str)   : The path to save application's outputs

    Notes
        1. You must provide either an existing configuration file or configuration settings. If both are provided, only the filename will be taken into account. This way the configuration settings will be ignored to construct the instance from the settings of the given file.
    """
    def __init__(
        self,
        p_filename: str = None,
        p_app_env: str = None,
        p_app_debug: str = None,
        p_fs_workspace: str = None,
        p_fs_weightsdir: str = None,
        p_fs_datadir: str = None,
        p_fs_resultsdir: str = None,
        p_fs_logsdir: str = None,
    ):
        # hard coded variables
        self.fext = ".ini"

        if p_filename is not None:
            self._read_from_existing_file(p_filename)
        else:
            if not [x for x in (p_app_env, p_app_debug, p_fs_workspace, p_fs_weightsdir, p_fs_datadir, p_fs_resultsdir) if x is None]:
                self.environement   = p_app_env
                self.debug          = p_app_debug

                self.workspace      = p_fs_workspace
                self.weights_dir    = p_fs_weightsdir
                self.data_dir       = p_fs_datadir
                self.result_dir     = p_fs_resultsdir
                self.log_dir       = p_fs_logsdir

                self._parse()
            else:
                raise ValueError("You must provide either an existing configuration file or configuration parameters.")

    def _parse(self):
        """
        Parse member variables into configparser.ConfigParser object.
        """
        self.config_parser = configparser.ConfigParser()

        # Sections
        self.config_parser.add_section("APP")
        self.config_parser.add_section("PATHS")

        # Settings
        self.config_parser.set("APP", 'ENVIRONMENT', self.environement)
        self.config_parser.set("APP", 'DEBUG', self.debug)

        self.config_parser.set("PATHS", "WORKSPACE", self.workspace)
        self.config_parser.set("PATHS", "WEIGHTS_DIR", self.weights_dir)
        self.config_parser.set("PATHS", "DATA_DIR", self.data_dir)
        self.config_parser.set("PATHS", "RESULT_DIR", self.result_dir)
        self.config_parser.set("PATHS", "LOG_DIR", self.log_dir)

    def _read_from_existing_file(self, filename):
        """
        Read an existing configuration file to populate member variables.

        Parameters
            filename (str) : The full path (<path>/<filename>) to the existing configuration file. 
        """
        if os.path.isfile(filename):
            self.config_parser = configparser.ConfigParser()
            self.config_parser.read(filename)

            self.environement   = self.config_parser["APP"]["ENVIRONMENT"]
            self.debug          = self.config_parser["APP"]["DEBUG"]   

            self.workspace      = self.config_parser["PATHS"]["WORKSPACE"]    
            self.weights_dir    = self.config_parser["PATHS"]["WEIGHTS_DIR"]
            self.data_dir       = self.config_parser["PATHS"]["DATA_DIR"]
            self.result_dir     = self.config_parser["PATHS"]["RESULT_DIR"]
            self.log_dir        = self.config_parser["PATHS"]["LOG_DIR"]
        else:
            raise FileNotFoundError("Configuration file not found : {}".format(filename))

    def save(self, filename:str):
        """
        Save the configuration into a file.

        Parameters
            filename      (str)   : The full path (<path>/<filename>) to save the configuration file. The extension '.ini' is set automatically. 
        """
        self._filename = filename

        # Save the file
        with open(self._filename + self.fext, 'w') as config_file:
            self.config_parser.write(config_file)
            config_file.flush()

        log.info("The `{}` configuration has been created".format(self._filename + self.fext))

    def __repr__(self): 
        content = ""

        for section in self.config_parser.sections():
            content = "".join([content, "[{}]\n".format(section)])
            for param in self.config_parser[section]:
                content = "".join([content, "\t{} : {}\n".format(param, self.config_parser[section][param])])

        return content

class CDefaultConfiguration(CConfiguration):
    """
    Class containing the default configuration of the application.
    As a subclass of CConfiguration, this class only create an instance of CConfiguration with default settings.
    """
    def __init__(self):
        workspace = os.getcwd()

        super().__init__(
            p_app_env="Dev",
            p_app_debug="True", 
            p_fs_workspace=workspace, 
            p_fs_weightsdir=os.path.join(workspace, "resources/weights"),
            p_fs_datadir=os.path.join(workspace, "resources/data"),
            p_fs_resultsdir=os.path.join(workspace, "results"),
            p_fs_logsdir=os.path.join(workspace, "logs")
        )

    def save(self):
        super().save("default")

if __name__=="__main__":
    default_cfg = CDefaultConfiguration()
    default_cfg.save()