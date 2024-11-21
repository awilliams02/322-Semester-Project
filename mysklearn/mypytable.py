##############################################
# Programmer: Alexa Williams
# Class: CptS 322-01, Fall 2024
# Programming Assignment #7
# 11/14/2024
# 
# Description: This file contains the 
#   MyPyTable class
##############################################

import copy
import csv
from tabulate import tabulate
from mysklearn import myutils

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.data[0])

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        if type(col_identifier) is str:
            col_index = self.column_names.index(col_identifier)
        else:
            col_index = col_identifier
        col = []
        for row in self.data:
            if include_missing_values == False:
                if row[col_index] != "NA":
                    col.append(row[col_index])
            else:
                col.append(row[col_index])
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    numeric_val = float(self.data[i][j])
                    self.data[i][j] = numeric_val
                except ValueError as e:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop.sort(reverse = True)
        for index in row_indexes_to_drop:
            if index < len(self.data):
                self.data.pop(index)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        infile = open(filename, "r")
        reader = csv.reader(infile)
        for row in reader:
            self.data.append(row)
        self.convert_to_numeric()
        infile.close()

        self.column_names = self.data[0]
        self.data.pop(0)

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        writer = csv.writer(outfile)
        writer.writerow(self.column_names)
        writer.writerows(self.data)
        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        duplicate_indeces = []
        key_indeces = [self.column_names.index(col) for col in key_column_names]
        seen_rows = []
        for i, row in enumerate(self.data):
            row_key = tuple(row[index] for index in key_indeces)
            if row_key in seen_rows:
                duplicate_indeces.append(i)
            else:
                seen_rows.append(row_key)
        return duplicate_indeces

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        self.data =[row for row in self.data if "NA" not in row]
        
    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        values = []
        for row in self.data:
            if row[col_index] != "NA":
                try:
                    values.append(float(row[col_index]))
                except ValueError:
                    pass
        
        if values:
            col_average = sum(values) / len(values)
            for row in self.data:
                if row[col_index] == "NA":
                    row[col_index] = col_average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        summary_data = []
        for col_name in col_names:
            col_index = self.column_names.index(col_name)

            values = []

            for row in self.data:
                if row[col_index] != "NA":
                    try:
                        values.append(float(row[col_index]))
                    except ValueError:
                        pass
            
            if values: 
                values.sort()
                col_min = min(values)
                col_max = max(values)
                col_mid = (col_min + col_max) / 2
                col_avg = sum(values) / len(values)
                n = len(values)
                if n % 2 == 1:
                    col_median = values[n // 2]
                else:
                    m1 = values[(n // 2 ) - 1]
                    m2 = values[(n // 2)]
                    col_median = (m1 + m2) / 2

                summary_data.append([col_name, col_min, col_max, col_mid, col_avg, col_median])
        
        summary_column_names = ["attribute", "min", "max", "mid", "avg", "median"]
        return MyPyTable(summary_column_names, summary_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        joined_table = []
        self_key_indeces = [self.column_names.index(col) for col in key_column_names]
        other_key_indeces = [other_table.column_names.index(col) for col in key_column_names]

        non_key_columns_other = [col for col in other_table.column_names if col not in key_column_names]
        non_key_indexes_other = [other_table.column_names.index(col) for col in non_key_columns_other]
        
        new_column_names = self.column_names + non_key_columns_other

        for row_self in self.data:
            key_self = tuple(row_self[i] for i in self_key_indeces)

            for row_other in other_table.data:
                key_other = tuple(row_other[i] for i in other_key_indeces)
                if key_self == key_other:
                    combined_row = row_self + [row_other[i] for i in non_key_indexes_other]
                    joined_table.append(combined_row)

        return MyPyTable(new_column_names, joined_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        joined_table = []
        self_key_indeces = [self.column_names.index(col) for col in key_column_names]
        other_key_indeces = [other_table.column_names.index(col) for col in key_column_names]
        non_key_columns_self = [col for col in self.column_names if col not in key_column_names]
        non_key_columns_other = [col for col in other_table.column_names if col not in key_column_names]
        non_key_indexes_other = [other_table.column_names.index(col) for col in non_key_columns_other]
        non_key_indexes_self = [self.column_names.index(col) for col in non_key_columns_self]
        
        new_column_names = self.column_names + non_key_columns_other
        joined_key_indeces = [new_column_names.index(col) for col in key_column_names]

        for row_self in self.data:
            key_self = tuple(row_self[i] for i in self_key_indeces)
            flag = False
            for row_other in other_table.data:
                key_other = tuple(row_other[i] for i in other_key_indeces)
                if key_self == key_other:
                    combined_row = row_self + [row_other[i] for i in non_key_indexes_other]
                    joined_table.append(combined_row)
                    flag = True
            if flag == False:
                combined_row = row_self + ["NA"] * len(non_key_columns_other)
                joined_table.append(combined_row)
        
        joined_table_copy = copy.copy(joined_table)
        for row_other in other_table.data:
            key_other = tuple(row_other[i] for i in other_key_indeces)
            flag = False
            for row_joined in joined_table_copy:
                key_joined = tuple(row_joined[i] for i in joined_key_indeces)
                if key_other == key_joined:
                    flag = True
            if flag == False:
                self_row = ["NA"] * len(self.column_names)
                j = 0
                for i in self_key_indeces:
                    self_row[i] = row_other[other_key_indeces[j]]
                    j = j + 1
                combined_row = self_row + [row_other[i] for i in non_key_indexes_other]
                joined_table.append(combined_row)

        return MyPyTable(new_column_names, joined_table)

