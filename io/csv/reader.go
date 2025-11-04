// Package csv provides CSV reading and writing functionality for DataFrames.
package csv

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	"github.com/TIVerse/GopherData/series"
)

// CSVReader reads CSV files into DataFrames.
type CSVReader struct {
	path      string
	delimiter rune
	header    bool
	naValues  []string
	chunkSize int
	parallel  int
	dtypes    map[string]core.Dtype
}

// CSVOption is a functional option for configuring CSVReader.
type CSVOption func(*CSVReader) error

// WithDelimiter sets the field delimiter (default: comma).
func WithDelimiter(delim rune) CSVOption {
	return func(r *CSVReader) error {
		r.delimiter = delim
		return nil
	}
}

// WithHeader specifies if the first row contains column names (default: true).
func WithHeader(header bool) CSVOption {
	return func(r *CSVReader) error {
		r.header = header
		return nil
	}
}

// WithNA sets the list of strings to treat as null values.
func WithNA(naValues []string) CSVOption {
	return func(r *CSVReader) error {
		r.naValues = naValues
		return nil
	}
}

// WithChunkSize sets the chunk size for reading large files.
func WithChunkSize(size int) CSVOption {
	return func(r *CSVReader) error {
		r.chunkSize = size
		return nil
	}
}

// WithParallel sets the number of parallel workers (default: runtime.NumCPU()).
func WithParallel(n int) CSVOption {
	return func(r *CSVReader) error {
		r.parallel = n
		return nil
	}
}

// WithDtypes sets explicit data types for columns.
func WithDtypes(dtypes map[string]core.Dtype) CSVOption {
	return func(r *CSVReader) error {
		r.dtypes = dtypes
		return nil
	}
}

// ReadCSV reads a CSV file and returns a DataFrame.
func ReadCSV(path string, opts ...CSVOption) (*dataframe.DataFrame, error) {
	reader := &CSVReader{
		path:      path,
		delimiter: ',',
		header:    true,
		naValues:  core.DefaultNAValues,
		chunkSize: 0,
		parallel:  0,
		dtypes:    nil,
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(reader); err != nil {
			return nil, err
		}
	}

	return reader.read()
}

// read performs the actual CSV reading.
func (r *CSVReader) read() (*dataframe.DataFrame, error) {
	file, err := os.Open(r.path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	csvReader := csv.NewReader(file)
	csvReader.Comma = r.delimiter
	csvReader.ReuseRecord = true

	// Read header or generate column names
	var columns []string
	if r.header {
		header, err := csvReader.Read()
		if err != nil {
			return nil, fmt.Errorf("failed to read header: %w", err)
		}
		columns = make([]string, len(header))
		copy(columns, header)
	}

	// Read all records
	var records [][]string
	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read record: %w", err)
		}

		// Make a copy since ReuseRecord is true
		recordCopy := make([]string, len(record))
		copy(recordCopy, record)
		records = append(records, recordCopy)
	}

	if len(records) == 0 {
		return dataframe.New(map[string]any{})
	}

	// Generate column names if no header
	ncols := len(records[0])
	if !r.header {
		columns = make([]string, ncols)
		for i := 0; i < ncols; i++ {
			columns[i] = fmt.Sprintf("col_%d", i)
		}
	}

	// Collect column values as strings first
	rawData := make(map[string][]string, len(columns))
	for i, col := range columns {
		rawData[col] = make([]string, len(records))
		for j, record := range records {
			if i < len(record) {
				rawData[col][j] = record[i]
			} else {
				rawData[col][j] = ""
			}
		}
	}

	// Infer types and parse data
	columnData := make(map[string]any, len(columns))

	// Infer types and parse
	for _, col := range columns {
		var dtype core.Dtype
		if r.dtypes != nil {
			if dt, exists := r.dtypes[col]; exists {
				dtype = dt
			} else {
				dtype = r.inferType(rawData[col])
			}
		} else {
			dtype = r.inferType(rawData[col])
		}

		parsed, err := r.parseColumn(rawData[col], dtype)
		if err != nil {
			return nil, fmt.Errorf("failed to parse column %q: %w", col, err)
		}
		columnData[col] = parsed
	}

	// Create DataFrame
	df, err := dataframe.New(columnData)
	if err != nil {
		return nil, fmt.Errorf("failed to create dataframe: %w", err)
	}

	// Mark null values
	for col, values := range rawData {
		s, _ := df.Column(col)
		for i, val := range values {
			if r.isNA(val) {
				s.SetNull(i)
			}
		}
	}

	return df, nil
}

// inferType infers the data type from a column of string values.
func (r *CSVReader) inferType(values []string) core.Dtype {
	hasInt := true
	hasFloat := true
	hasBool := true

	for _, val := range values {
		if r.isNA(val) {
			continue
		}

		// Try int
		if hasInt {
			if _, err := strconv.ParseInt(val, 10, 64); err != nil {
				hasInt = false
			}
		}

		// Try float
		if hasFloat {
			if _, err := strconv.ParseFloat(val, 64); err != nil {
				hasFloat = false
			}
		}

		// Try bool
		if hasBool {
			lower := strings.ToLower(val)
			if lower != "true" && lower != "false" {
				hasBool = false
			}
		}

		// If nothing matches, it's a string
		if !hasInt && !hasFloat && !hasBool {
			return core.DtypeString
		}
	}

	if hasBool {
		return core.DtypeBool
	}
	if hasInt {
		return core.DtypeInt64
	}
	if hasFloat {
		return core.DtypeFloat64
	}
	return core.DtypeString
}

// parseColumn parses a column of string values into the specified type.
func (r *CSVReader) parseColumn(values []string, dtype core.Dtype) ([]any, error) {
	result := make([]any, len(values))

	for i, val := range values {
		if r.isNA(val) {
			// Will be marked as null later
			switch dtype {
			case core.DtypeInt64:
				result[i] = int64(0)
			case core.DtypeFloat64:
				result[i] = float64(0)
			case core.DtypeBool:
				result[i] = false
			case core.DtypeString:
				result[i] = ""
			default:
				result[i] = ""
			}
			continue
		}

		switch dtype {
		case core.DtypeInt64:
			parsed, err := strconv.ParseInt(val, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("failed to parse %q as int64: %w", val, err)
			}
			result[i] = parsed

		case core.DtypeFloat64:
			parsed, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, fmt.Errorf("failed to parse %q as float64: %w", val, err)
			}
			result[i] = parsed

		case core.DtypeBool:
			lower := strings.ToLower(val)
			if lower == "true" {
				result[i] = true
			} else if lower == "false" {
				result[i] = false
			} else {
				return nil, fmt.Errorf("failed to parse %q as bool", val)
			}

		case core.DtypeString:
			result[i] = val

		default:
			result[i] = val
		}
	}

	return result, nil
}

// isNA checks if a value should be treated as null/NA.
func (r *CSVReader) isNA(val string) bool {
	for _, na := range r.naValues {
		if val == na {
			return true
		}
	}
	return false
}

// ReadCSVToSeries reads a single column from a CSV file as a Series.
func ReadCSVToSeries(path string, colName string, opts ...CSVOption) (*series.Series[any], error) {
	df, err := ReadCSV(path, opts...)
	if err != nil {
		return nil, err
	}

	return df.Column(colName)
}
