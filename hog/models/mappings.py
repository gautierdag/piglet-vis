import pickle
from typing import Dict

OBJECT_ATTRIBUTES = [
    "ObjectName",
    "parentReceptacles",
    "receptacleObjectIds",
    "distance",
    "mass",
    "size",
    "ObjectTemperature",
    "breakable",
    "cookable",
    "dirtyable",
    "isBroken",
    "isCooked",
    "isDirty",
    "isFilledWithLiquid",
    "isOpen",
    "isPickedUp",
    "isSliced",
    "isToggled",
    "moveable",
    "openable",
    "pickupable",
    "receptacle",
    "salientMaterials_Ceramic",
    "salientMaterials_Fabric",
    "salientMaterials_Food",
    "salientMaterials_Glass",
    "salientMaterials_Leather",
    "salientMaterials_Metal",
    "salientMaterials_Paper",
    "salientMaterials_Plastic",
    "salientMaterials_Rubber",
    "salientMaterials_Soap",
    "salientMaterials_Sponge",
    "salientMaterials_Stone",
    "salientMaterials_Wax",
    "salientMaterials_Wood",
    "sliceable",
    "toggleable",
]


def get_actions_mapper(data_dir_path: str) -> Dict[int, str]:
    """
    Return a dict mapping index to action name
    """
    with open(f"{data_dir_path}/reverse_action_mapping.pkl", "rb") as f:
        return pickle.load(f)


def get_objects_mapper(data_dir_path: str) -> Dict[int, Dict[int, str]]:
    """
    Return a dict mapping index to object name
    """
    with open(f"{data_dir_path}/reverse_object_mapping.pkl", "rb") as f:
        # select first index since the first index is the objet name encoded as an int
        return pickle.load(f)
