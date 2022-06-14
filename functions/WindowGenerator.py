import time
from typing import Tuple
import numpy as np
import pandas
import tqdm
from constants import (
    OOV_TOKEN,
    REARRANGED_SINGLE_INPUT_WINDOWED_DATA_FILEPATH,
    REARRANGED_SINGLE_INPUT_WINDOWED_LABEL_FILEPATH,
)
from functions.clamp import clamp
import pickle


class WindowGenerator:
    """
    Class to generate timestep'd data
    """

    def __init__(self, input_width: int, output_width: int, save_windows: bool):
        """Init Parmas

        Args:
            input_width (int): The timesteps forming the input sequence
            output_width (int): The timesteps forming the output sequence
        """
        self.input_width: int = input_width
        self.output_width: int = output_width
        self.total_window_size: int = input_width + output_width
        self.minimum_day_of_year: int = 0
        self.maximum_day_of_year: int = 365
        self.save_windows: bool = save_windows

    def split_window(self, data: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

        all_labels: list = []
        all_input_timestep_sequences: list = []
        non_null_indexes = list(
            zip(*np.where(data.notnull()))
        )  # Get indexes of df where values which are not null
        for i in tqdm.tqdm(range(len(data.index)), ncols=100, desc="Windowing data..."):
            input_sequence: list = []  # Input sequence per patient
            visit_indexes = [
                item[1] for item in non_null_indexes if item[0] == i
            ]  # Indexes of all EHR entries for patient
            for visit_index in visit_indexes:
                input_timesteps_sequence: list = []
                lower_bound = clamp(
                    self.minimum_day_of_year,
                    visit_index - self.input_width,
                    self.maximum_day_of_year,
                )
                upper_bound = clamp(
                    self.minimum_day_of_year,
                    visit_index + self.output_width,
                    self.maximum_day_of_year,
                )
                row_slice = data.iloc[[i]]
                input_sequence = row_slice.iloc[:, lower_bound + 1 : visit_index + 1]
                print("")
                label_sequence = row_slice.iloc[:, visit_index + 1 : upper_bound]

                all_labels.append(self._categorize_output_sequence(label_sequence))

                for column in input_sequence:
                    if input_sequence[column].isnull().values.any():
                        input_timesteps_sequence.append(OOV_TOKEN)
                    else:
                        input_timesteps_sequence.append(
                            ",".join(str(x) for x in input_sequence[column].values)
                        )

                # check that the input sequence matches the input_width
                if len(input_timesteps_sequence) != self.input_width:
                    # find difference in length and pad from the left of the input sequence
                    diff = self.input_width - len(input_timesteps_sequence)
                    for _ in range(diff):
                        input_timesteps_sequence.insert(0, OOV_TOKEN)

                all_input_timestep_sequences.append(input_timesteps_sequence)

        return all_input_timestep_sequences, all_labels

    def split_single_input_sequence(self, data: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a single input sequence for prediction
        Only the clincial text of the last visit forms input

        Output - Labels:  Binary classification of output width (0 no revisit, 1 revisit)
               - Input Sequences: Textual input sequences
        Args:
            data (pandas.DataFrame): Dataframe with reorganised data.
            save_windows (bool, optional): Save windows variables to file. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Input Sequences, Labels
        """
        
        data = data.head(100)
        non_null_indexes = list(
            zip(*np.where(data.notnull()))
        )  # Get indexes of df where values which are not null

        all_labels: list = []
        input_timesteps_sequence: list = []

        for i in tqdm.tqdm(range(len(data.index)), ncols=100, desc="Windowing data..."):
            visit_indexes = [
                item[1] for item in non_null_indexes if item[0] == i
            ]  # Indexes of all EHR entries for patient
            for visit_index in visit_indexes:

                upper_bound = clamp(
                    self.minimum_day_of_year,
                    visit_index + self.output_width,
                    self.maximum_day_of_year,
                )

                input_timesteps_sequence.append(data.iat[i, visit_index])
                label_sequence = data.iloc[i:, visit_index + 1 : upper_bound]
                all_labels.append(self._categorize_output_sequence(label_sequence))

        if self.save_windows:
            print("------Saving windows for reuse ------")
            with open(REARRANGED_SINGLE_INPUT_WINDOWED_DATA_FILEPATH, "wb") as f:
                pickle.dump(input_timesteps_sequence, f)
            with open(REARRANGED_SINGLE_INPUT_WINDOWED_LABEL_FILEPATH, "wb") as f:
                pickle.dump(all_labels, f)
        return input_timesteps_sequence, all_labels

    def _categorize_output_sequence(self, output_sequence: pandas.DataFrame) -> bool:
        """Categorise output sequence to binary
        Classification is based on if output sequence is not null in the output width
        Args:
            output_sequence (pandas.DataFrame): Sequence to classify

        Returns:
            bool: 0 = no revisit, 1 = revisit
        """
        if output_sequence.isnull().all().all():
            return 0
        else:
            return 1
