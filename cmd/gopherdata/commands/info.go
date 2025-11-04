package commands

import (
	"fmt"

	"github.com/TIVerse/GopherData/dataframe"
	"github.com/TIVerse/GopherData/io/csv"
	"github.com/spf13/cobra"
)

// InfoCmd returns the info command.
func InfoCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "info <file>",
		Short: "Show file metadata and structure",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			df, err := csv.ReadCSV(args[0])
			if err != nil {
				return fmt.Errorf("failed to read file: %w", err)
			}

			fmt.Printf("File: %s\n", args[0])
			fmt.Printf("Rows: %d\n", df.Nrows())
			fmt.Printf("Columns: %d\n", df.Ncols())
			fmt.Printf("\nColumn Names:\n")
			for i, col := range df.Columns() {
				series, _ := df.Column(col)
				fmt.Printf("  %d. %s (%s)\n", i+1, col, series.Dtype())
			}

			return nil
		},
	}
}

// HeadCmd returns the head command.
func HeadCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "head <file>",
		Short: "Show first N rows",
		Args:  cobra.ExactArgs(1),
	}

	n := cmd.Flags().IntP("rows", "n", 10, "Number of rows to display")

	cmd.RunE = func(cmd *cobra.Command, args []string) error {
		df, err := csv.ReadCSV(args[0])
		if err != nil {
			return fmt.Errorf("failed to read file: %w", err)
		}

		rows := *n
		if rows > df.Nrows() {
			rows = df.Nrows()
		}

		// Print header
		cols := df.Columns()
		for i, col := range cols {
			if i > 0 {
				fmt.Print("\t")
			}
			fmt.Print(col)
		}
		fmt.Println()

		// Print rows
		for i := 0; i < rows; i++ {
			for j, col := range cols {
				series, _ := df.Column(col)
				val, _ := series.Get(i)
				if j > 0 {
					fmt.Print("\t")
				}
				fmt.Print(val)
			}
			fmt.Println()
		}

		return nil
	}

	return cmd
}

// TailCmd returns the tail command.
func TailCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "tail <file>",
		Short: "Show last N rows",
		Args:  cobra.ExactArgs(1),
	}

	n := cmd.Flags().IntP("rows", "n", 10, "Number of rows to display")

	cmd.RunE = func(cmd *cobra.Command, args []string) error {
		df, err := csv.ReadCSV(args[0])
		if err != nil {
			return fmt.Errorf("failed to read file: %w", err)
		}

		rows := *n
		if rows > df.Nrows() {
			rows = df.Nrows()
		}

		start := df.Nrows() - rows

		// Print header
		cols := df.Columns()
		for i, col := range cols {
			if i > 0 {
				fmt.Print("\t")
			}
			fmt.Print(col)
		}
		fmt.Println()

		// Print rows
		for i := start; i < df.Nrows(); i++ {
			for j, col := range cols {
				series, _ := df.Column(col)
				val, _ := series.Get(i)
				if j > 0 {
					fmt.Print("\t")
				}
				fmt.Print(val)
			}
			fmt.Println()
		}

		return nil
	}

	return cmd
}

// DescribeCmd returns the describe command.
func DescribeCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "describe <file>",
		Short: "Show statistical summary",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			df, err := csv.ReadCSV(args[0])
			if err != nil {
				return fmt.Errorf("failed to read file: %w", err)
			}

			fmt.Printf("Statistical Summary for: %s\n\n", args[0])
			fmt.Printf("Shape: %d rows x %d columns\n\n", df.Nrows(), df.Ncols())

			// For each numeric column, show stats
			for _, col := range df.Columns() {
				series, _ := df.Column(col)
				if series.Dtype() == 1 { // Float64
					fmt.Printf("Column: %s\n", col)
					fmt.Printf("  Type: %s\n", series.Dtype())
					fmt.Printf("  Count: %d\n", series.Len())
					// Could add mean, std, etc. here
					fmt.Println()
				}
			}

			return nil
		},
	}
}

// ConvertCmd returns the convert command.
func ConvertCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "convert <input> <output>",
		Short: "Convert between data formats",
		Long:  "Convert data files between formats (CSV, JSON, etc.)",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			input := args[0]
			output := args[1]

			// Read input (assume CSV for now)
			df, err := csv.ReadCSV(input)
			if err != nil {
				return fmt.Errorf("failed to read input: %w", err)
			}

			// Write output (assume CSV for now)
			err = csv.ToCSV(df, output)
			if err != nil {
				return fmt.Errorf("failed to write output: %w", err)
			}

			fmt.Printf("Converted %s -> %s\n", input, output)
			fmt.Printf("Rows: %d, Columns: %d\n", df.Nrows(), df.Ncols())

			return nil
		},
	}
}

// FilterCmd returns the filter command.
func FilterCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "filter <file> <output>",
		Short: "Filter rows based on conditions",
		Args:  cobra.ExactArgs(2),
	}

	column := cmd.Flags().StringP("column", "c", "", "Column name")
	value := cmd.Flags().StringP("value", "v", "", "Value to filter by")

	cmd.RunE = func(cmd *cobra.Command, args []string) error {
		if *column == "" || *value == "" {
			return fmt.Errorf("both --column and --value are required")
		}

		df, err := csv.ReadCSV(args[0])
		if err != nil {
			return err
		}

		// Simple string match filter
		filtered := df.Filter(func(row *dataframe.Row) bool {
			val, ok := row.Get(*column)
			if !ok {
				return false
			}
			return fmt.Sprint(val) == *value
		})

		err = csv.ToCSV(filtered, args[1])
		if err != nil {
			return err
		}

		fmt.Printf("Filtered %d -> %d rows\n", df.Nrows(), filtered.Nrows())
		return nil
	}

	return cmd
}

// SelectCmd returns the select command.
func SelectCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "select <file> <output>",
		Short: "Select specific columns",
		Args:  cobra.ExactArgs(2),
	}

	columns := cmd.Flags().StringSliceP("columns", "c", nil, "Columns to select (comma-separated)")

	cmd.RunE = func(cmd *cobra.Command, args []string) error {
		if len(*columns) == 0 {
			return fmt.Errorf("--columns is required")
		}

		df, err := csv.ReadCSV(args[0])
		if err != nil {
			return err
		}

		selected := df.Select(*columns...)

		err = csv.ToCSV(selected, args[1])
		if err != nil {
			return err
		}

		fmt.Printf("Selected %d columns from %d\n", len(*columns), df.Ncols())
		return nil
	}

	return cmd
}
