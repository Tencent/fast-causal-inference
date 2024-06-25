import os


def get_user():
    if str(os.environ.get("JUPYTERHUB_USER")) != "None":
        return str(os.environ.get("JUPYTERHUB_USER"))
    elif str(os.environ.get("USER")) != "None":
        return str(os.environ.get("USER"))
    elif str(os.environ.get("CURRENT_USER")) != "None":
        return str(os.environ.get("CURRENT_USER"))
    else:
        return str("default")
