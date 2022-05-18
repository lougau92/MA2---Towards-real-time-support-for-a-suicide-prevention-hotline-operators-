from typing import Dict
import yaml

from ..model.base_model import BaseModel


class Config:
    def __init__(self) -> None:
        self.__model_register = {}

    def register_model(self, name: str, model_class: BaseModel) -> None:
        """Method registers a model corresponding to the provided name.

        Args:
            name (str): The name, so the model can be accessed later on.
            model_class (BaseModel): The class of the model to register.
        """
        self.__model_register[name] = model_class

    def get_model(self, name: str) -> BaseModel:
        """Method returns a registered BaseModel.

        Args:
            name (str): The models register name

        Raises:
            RuntimeError: Occurs if the model was not registered.

        Returns:
            BaseModel: The model corresponding to the name.
        """
        model = self.__model_register.get(name, None)
        if not model:
            raise RuntimeError(f"Model of type {name} is not registered.")

        return model

    def get_args(self, path: str) -> Dict:
        """This method reads the hyperparameters and program args of a 
        specified yaml and returns them as dictionary.

        Args:
            path (str): The yaml path file.

        Returns:
            Dict: The dictionary containing all arguments.
        """
        with open(path, "r") as stream:
            args = yaml.safe_load(stream)

        return args

    def store_args(self, path: str, args: Dict) -> None:
        with open(path, "w") as stream:
            yaml.safe_dump(args, stream)


config = Config()