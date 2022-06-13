import numpy as np
import tqdm
from constants import OOV_TOKEN
from functions.clamp import clamp


class WindowGenerator:
    """Class to generate timestep data"""

    def __init__(self, input_width: int, output_width: int):
        self.input_width: int = input_width
        self.output_width: int = output_width
        self.total_window_size: int = input_width + output_width
        self.minimum_day_of_year: int = 0
        self.maximum_day_of_year: int = 365

    def split_window(self, data):
        """Creates split dataset

        Args:
            data (pan): _description_
        Returns:
            _type_: _description_
        """
        all_labels: list = []
        all_input_timestep_sequences: list = []
        data = data.head(10)
        data = data.replace(r"^\s*$", np.nan, regex=True)
        non_null_indexes = list(
            zip(*np.where(data.notnull()))
        )  # Get indexes of df where values which are not null
        newdf = data.notnull().astype("int")
        print(newdf.head())
        for i in tqdm.tqdm(range(len(data.index)), ncols=100, desc="Windowing data.."):
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

                if label_sequence.isnull().all().all():
                    all_labels.append(0)
                else:
                    all_labels.append(1)
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
