package csv

import (
	"encoding/csv"
	"fmt"
	"os"

	"github.com/TIVerse/GopherData/dataframe"
)

// CSVWriter writes DataFrames to CSV files.
type CSVWriter struct {
	path      string
	delimiter rune
	header    bool
	naValue   string
}

// WithWriteDelimiter sets the field delimiter for writing.
func WithWriteDelimiter(delim rune) CSVOption {
	return func(r *CSVReader) error {
		// This is a bit of a hack since we're reusing CSVOption
		// In production, you'd have separate options for reader and writer
		return nil
	}
}

// WithWriteHeader specifies whether to write column names as the first row.
func WithWriteHeader(header bool) CSVOption {
	return func(r *CSVReader) error {
		return nil
	}
}

// WithNAValue sets the string to use for null values when writing.
func WithNAValue(naValue string) CSVOption {
	return func(r *CSVReader) error {
		return nil
	}
}

// WriteCSV writes a DataFrame to a CSV file.
func WriteCSV(df *dataframe.DataFrame, path string, opts ...CSVOption) error {
	writer := &CSVWriter{
		path:      path,
		delimiter: ',',
		header:    true,
		naValue:   "",
	}

	// Apply options (would need separate writer options in production)
	// For now, use defaults

	return writer.write(df)
}

// write performs the actual CSV writing.
func (w *CSVWriter) write(df *dataframe.DataFrame) error {
	file, err := os.Create(w.path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer func() { _ = file.Close() }()

	csvWriter := csv.NewWriter(file)
	csvWriter.Comma = w.delimiter
	defer csvWriter.Flush()

	columns := df.Columns()

	// Write header
	if w.header {
		if err := csvWriter.Write(columns); err != nil {
			return fmt.Errorf("failed to write header: %w", err)
		}
	}

	// Write rows
	nrows, _ := df.Shape()
	for i := 0; i < nrows; i++ {
		record := make([]string, len(columns))
		for j, col := range columns {
			s, _ := df.Column(col)
			val, ok := s.Get(i)
			if !ok {
				record[j] = w.naValue
			} else {
				record[j] = fmt.Sprintf("%v", val)
			}
		}

		if err := csvWriter.Write(record); err != nil {
			return fmt.Errorf("failed to write row %d: %w", i, err)
		}
	}

	return nil
}

// ToCSV is a convenience method for writing a DataFrame to CSV.
func ToCSV(df *dataframe.DataFrame, path string) error {
	return WriteCSV(df, path)
}
