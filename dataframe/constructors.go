package dataframe

import (
	"fmt"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/series"
)

// New creates a new DataFrame from a map of column names to data slices.
// All slices must have the same length.
// Automatically infers data types and creates a default RangeIndex.
func New(data map[string]any) (*DataFrame, error) {
	if len(data) == 0 {
		return &DataFrame{
			columns: []string{},
			series:  make(map[string]*series.Series[any]),
			nrows:   0,
		}, nil
	}

	// Extract column names and determine number of rows
	columns := make([]string, 0, len(data))
	nrows := -1

	for col, values := range data {
		columns = append(columns, col)

		// Determine length
		var length int
		switch v := values.(type) {
		case []int64:
			length = len(v)
		case []float64:
			length = len(v)
		case []string:
			length = len(v)
		case []bool:
			length = len(v)
		case []any:
			length = len(v)
		default:
			return nil, fmt.Errorf("unsupported type for column %q: %T", col, values)
		}

		if nrows == -1 {
			nrows = length
		} else if length != nrows {
			return nil, fmt.Errorf("column %q has length %d, expected %d: %w",
				col, length, nrows, core.ErrInvalidShape)
		}
	}

	// Create series map
	seriesMap := make(map[string]*series.Series[any])

	for col, values := range data {
		var s *series.Series[any]

		switch v := values.(type) {
		case []int64:
			s = convertToAnySeries(col, v, core.DtypeInt64)
		case []float64:
			s = convertToAnySeries(col, v, core.DtypeFloat64)
		case []string:
			s = convertToAnySeries(col, v, core.DtypeString)
		case []bool:
			s = convertToAnySeries(col, v, core.DtypeBool)
		case []any:
			// Infer type from first non-nil element
			dtype := inferDtype(v)
			s = series.New(col, v, dtype)
		default:
			return nil, fmt.Errorf("unsupported type for column %q: %T", col, values)
		}

		seriesMap[col] = s
	}

	df := &DataFrame{
		columns: columns,
		series:  seriesMap,
		index:   NewRangeIndex(0, nrows, 1),
		nrows:   nrows,
	}

	return df, nil
}

// FromRecords creates a DataFrame from a slice of maps (records).
// Each map represents a row with column names as keys.
func FromRecords(records []map[string]any) (*DataFrame, error) {
	if len(records) == 0 {
		return &DataFrame{
			columns: []string{},
			series:  make(map[string]*series.Series[any]),
			nrows:   0,
		}, nil
	}

	// Collect all unique column names
	columnSet := make(map[string]bool)
	for _, record := range records {
		for col := range record {
			columnSet[col] = true
		}
	}

	columns := make([]string, 0, len(columnSet))
	for col := range columnSet {
		columns = append(columns, col)
	}

	// Build column data
	data := make(map[string][]any)
	for _, col := range columns {
		data[col] = make([]any, len(records))
	}

	for i, record := range records {
		for _, col := range columns {
			if val, exists := record[col]; exists {
				data[col][i] = val
			} else {
				data[col][i] = nil // Will be marked as null
			}
		}
	}

	// Create series map
	seriesMap := make(map[string]*series.Series[any])

	for _, col := range columns {
		dtype := inferDtype(data[col])
		s := series.New(col, data[col], dtype)

		// Mark nil values as null
		for i, val := range data[col] {
			if val == nil {
				s.SetNull(i)
			}
		}

		seriesMap[col] = s
	}

	df := &DataFrame{
		columns: columns,
		series:  seriesMap,
		index:   NewRangeIndex(0, len(records), 1),
		nrows:   len(records),
	}

	return df, nil
}

// convertToAnySeries converts a typed slice to a Series[any].
func convertToAnySeries[T any](name string, data []T, dtype core.Dtype) *series.Series[any] {
	anyData := make([]any, len(data))
	for i, v := range data {
		anyData[i] = v
	}
	return series.New(name, anyData, dtype)
}

// inferDtype infers the Dtype from a slice of any values.
func inferDtype(values []any) core.Dtype {
	// Look at first non-nil value
	for _, v := range values {
		if v == nil {
			continue
		}

		switch v.(type) {
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			return core.DtypeInt64
		case float32, float64:
			return core.DtypeFloat64
		case bool:
			return core.DtypeBool
		case string:
			return core.DtypeString
		default:
			return core.DtypeString // Default to string
		}
	}

	// All nil, default to string
	return core.DtypeString
}
