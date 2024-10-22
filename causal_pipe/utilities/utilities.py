import json


def dump_json_to(data, path: str):
    """
    Dump data to a JSON file.

    Parameters:
    - data: Data to be dumped.
    - path (str): File path where the JSON will be saved.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)
