import time
from typing import Tuple
from black import out
import numpy as np
import pandas
import tqdm
from constants import (
    EMPTY_TIMESTEP_TOKEN,
    OOV_TOKEN,
    REARRANGED_MULTI_INPUT_WINDOWED_DATA_FILEPATH,
    REARRANGED_MULTI_INPUT_WINDOWED_LABEL_FILEPATH,
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

    def window_multi_input_sequence(
        self, data: pandas.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        sequence: list = []
        labels: list = []
        non_null_indexes = list(
            zip(*np.where(data.notnull()))
        )  # Get indexes of df where values which are not null

        for index in non_null_indexes:
            row = index[0]
            column = index[1]

            lower_bound = clamp(
                self.minimum_day_of_year,
                column - self.input_width,
                self.maximum_day_of_year,
            )

            upper_bound = clamp(
                self.minimum_day_of_year,
                column + self.input_width + 1,
                self.maximum_day_of_year,
            )

            input_sequence = data.iloc[row, lower_bound + 1 : column + 1]
            output_sequence = data.iloc[row, column + 1 : upper_bound]
            input_sequence = self._pad_timeseries(sequence=input_sequence)
            sequence.append(input_sequence)

            label = self._categorize_output_sequence(output_sequence=output_sequence)
            labels.append(label)
        if self.save_windows:
            self.save_frames(
                output_labels=np.array(labels), input_sequence=sequence, multi=True
            )

        return sequence, np.array(labels)

    def _pad_timeseries(self, sequence):
        pad_nan_delta = self.input_width - len(sequence)
        if pad_nan_delta > 0:
            sequence = np.pad(
                sequence,
                (pad_nan_delta, 0),
                "constant",
                constant_values=EMPTY_TIMESTEP_TOKEN,
            )
        return sequence

    def save_window(self, input_sequence_per_visit, output_category_per_visit):
        print("------Saving windows for reuse ------")
        with open(REARRANGED_SINGLE_INPUT_WINDOWED_DATA_FILEPATH, "wb") as f:
            pickle.dump(input_sequence_per_visit, f)
        with open(REARRANGED_SINGLE_INPUT_WINDOWED_LABEL_FILEPATH, "wb") as f:
            pickle.dump(output_category_per_visit, f)

    def _extract_inputs_outputs_sequence(
        self, row: pandas.DataFrame, visit_index: int
    ) -> Tuple[list, list]:
        """Creates a list of input sequence and output sequence based on width

        Args:
            row (pandas.DataFrame): row of the dataframe
            visit_index (int): when the visit occured, i.e DOY

        Returns:
            Tuple[list, list]: input sequence (len<=input width), output sequence (len<= output width)
        """
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
        input_sequence = row.iloc[:, lower_bound + 1 : visit_index + 1]
        output_sequence = row.iloc[:, visit_index + 1 : upper_bound]

        return np.array(input_sequence), np.array(output_sequence)

    def window_single_input_sequence(
        self, data: pandas.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
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
                label_sequence = data.iloc[i, visit_index + 1 : upper_bound]
                all_labels.append(self._categorize_output_sequence(label_sequence))

        if self.save_windows:
            self.save_frames(
                output_labels=all_labels, input_sequence=input_timesteps_sequence
            )
        return input_timesteps_sequence, all_labels

    def save_frames(self, output_labels, input_sequence, multi: bool = False):
        print("------Saving windows for reuse ------")
        if multi:
            with open(REARRANGED_MULTI_INPUT_WINDOWED_DATA_FILEPATH, "wb") as f:
                pickle.dump(input_sequence, f)
            with open(REARRANGED_MULTI_INPUT_WINDOWED_LABEL_FILEPATH, "wb") as f:
                pickle.dump(output_labels, f)
        else:
            with open(REARRANGED_SINGLE_INPUT_WINDOWED_DATA_FILEPATH, "wb") as f:
                pickle.dump(input_sequence, f)
            with open(REARRANGED_SINGLE_INPUT_WINDOWED_LABEL_FILEPATH, "wb") as f:
                pickle.dump(output_labels, f)

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
