from colorama import init, Fore, Back, Style


def print_green(info, value=""):
    """Utility to print a message in green

    Message will be print as [info] value

    Args:
        s (str): input string
        value (str): additional information to print

    Returns:
        (str): colored message
    """

    print(Fore.GREEN + "[%s] " % info + Style.RESET_ALL + str(value))


def print_red(info, value=""):
    """Utility to print a message in red

    Message will be print as [info] value

    Args:
        s (str): input string

    Returns:
        (str): colored message

    """

    print(Fore.RED + "[%s] " % info + Style.RESET_ALL + str(value))


def str_to_bluestr(s):
    """Utility to print a string in blue

    Args:
        s (str): input string

    Returns:
        (str): colored output string
    """

    return Fore.BLUE + "%s" % s + Style.RESET_ALL


def str_to_yellowstr(s):
    """Utility to print a string in yellow

    Args:
        s (str): input string

    Returns:
        (str): colored output string
    """

    return Fore.YELLOW + "%s" % s + Style.RESET_ALL


def str_to_brightstr(s):
    """Utility to print a string with heavy font

    Args:
        s (str): input string
    """

    print(Style.BRIGHT + s + Style.RESET_ALL)
